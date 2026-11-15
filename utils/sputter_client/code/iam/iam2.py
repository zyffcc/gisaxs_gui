'IAM v0.2: additions to iam.py (v0 kept unchanged for reproducibility).'

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import iam as M
from iam import (K0, RE, AU_F1_12, GH_X, GH_W, dc, rho_e, stack, layer_mid_int, jinc, s_para, lma_D, ff_sep, ff_arr,
                 tio2_layers, TIO2_STACK, R_PARTICLE, K_FL, EFF_RATIO, P_L3, MU_AU_12, OMEGA_L3)

COAT = os.environ.get("IAM_COAT", "bulk")


_TF = os.environ.get("IAM_TIO2_FIX", "0")
TIO2_FIX_A, TIO2_FIX_B = _TF in ("1", "a"), _TF in ("1", "b")
TIO2_FIX = TIO2_FIX_A or TIO2_FIX_B


import tio2_dwba as TD
TD.check_config(_TF == "1", M.TIO2_NUM)


CAP_PLACE = os.environ.get("IAM_CAP_PLACE", "in")
CAP_AVG = os.environ.get("IAM_CAP_AVG", "mid")
CAP_NEW = CAP_PLACE == "on" or CAP_AVG == "int"
if CAP_PLACE not in ("in", "on") or CAP_AVG not in ("mid", "int"):
    raise SystemExit("IAM_CAP_PLACE in|on, IAM_CAP_AVG mid|int")
if CAP_NEW and (_TF != "1" or os.environ.get("IAM_CAP", "0") != "1" or os.environ.get("IAM_CAPSCAT", "0") == "1"
                or os.environ.get("IAM_OZDEPTH", "0") == "1" or TD.DWBA != 0 or TD.NET != 0 or TD.CONTRAST != "vac"):
    raise SystemExit("IAM_CAP_PLACE=on / IAM_CAP_AVG=int need IAM_CAP=1 and IAM_TIO2_FIX=1; not with IAM_CAPSCAT, "
                     "IAM_OZDEPTH or the tio2_dwba switches")


TFILM = os.environ.get("IAM_TFILM", "0") == "1"
TFILM_N = float(os.environ.get("IAM_TFILM_N", "4"))
TFILM_AFR = float(os.environ.get("IAM_TFILM_AFR", "0.25"))


TFILM_T = os.environ.get("IAM_TFILM_T", "cap")
TFILM_PHI = float(os.environ.get("IAM_TFILM_PHI", "1"))
TFILM_KT = float(os.environ.get("IAM_TFILM_KT", "1"))
if TFILM_T not in ("cap", "hm", "fit"):
    raise SystemExit("IAM_TFILM_T cap|hm|fit")
if TFILM and TFILM_T == "cap" and not CAP_NEW:
    raise SystemExit("IAM_TFILM=1 needs the TOP-lite cap fields (IAM_CAP=1, IAM_CAP_PLACE=on or IAM_CAP_AVG=int)")
if TFILM and TFILM_T != "cap" and _TF != "1":
    raise SystemExit("IAM_TFILM_T=hm|fit needs IAM_TIO2_FIX=1")


TFILM_X = os.environ.get("IAM_TFILM_X", "rho")
TFILM_MET = os.environ.get("IAM_TFILM_MET", "0") == "1"
if TFILM_X not in ("rho", "fit"):
    raise SystemExit("IAM_TFILM_X rho|fit")
if (TFILM_X == "fit" or TFILM_MET) and not TFILM:
    raise SystemExit("IAM_TFILM_X=fit / IAM_TFILM_MET=1 need IAM_TFILM=1")
if TFILM_MET and (TFILM_T != "cap" or CAP_PLACE != "on"):
    raise SystemExit("IAM_TFILM_MET=1 needs IAM_TFILM_T=cap and IAM_CAP_PLACE=on")


def cap_avg_product(ai_amp, af_amp):
    '(1/t) int_0^t |E_i(z)|^2 |E_f(z)|^2 dz inside one layer (IAM_CAP_AVG=int), E(z) = A e^{i kz z} + Bb e^{-i kz (z-t)}'
    t = float(ai_amp[3])
    def terms(A, Bb, k):
        e = ((A, k, 0.0), (Bb, -k, t))
        return [(a * np.conj(b), ka - np.conj(kb), ka * za - np.conj(kb) * zb) for a, ka, za in e for b, kb, zb in e]
    out = 0.0
    for ci, li, fi in terms(*ai_amp[:3]):
        for cf, lf, ff in terms(*af_amp[:3]):
            x = 1j * (li + lf) * t
            e0 = np.exp(-1j * (fi + ff))
            small = np.abs(x) < 30.0
            xs = np.where(small, x, 1.0)
            xs = np.where(np.abs(xs) < 1e-12, 1e-12, xs)
            mid = np.where(np.abs(x) < 1e-12, 1.0 + 0.5 * x, np.expm1(xs) / xs)
            with np.errstate(over="ignore", invalid="ignore"):
                big = (np.exp(np.where(small, 0.0, x) - 1j * (fi + ff)) - e0) / np.where(small, 1.0, x)
            out = out + ci * cf * np.where(small, e0 * mid, big)
    return np.real(out)

_HD = np.load(Path(__file__).resolve().parent / "hd_table.npz")
HD_ETA, HD_K, HD_S = _HD["eta"], _HD["ks"], _HD["S"]
ETA_MAX = float(HD_ETA[-1])


def s_hd(qsig, eta):
    '2D hard-disk PY structure factor at q*sigma (array) and packing fraction eta (clamped to the table).'
    e = float(np.clip(eta, 0.0, ETA_MAX))
    i = min(int(np.searchsorted(HD_ETA, e)), len(HD_ETA) - 1)
    i0 = max(i - 1, 0)
    f = 0.0 if i == i0 else (e - HD_ETA[i0]) / (HD_ETA[i] - HD_ETA[i0])
    row = (1 - f) * HD_S[i0] + f * HD_S[i]
    return np.interp(qsig, HD_K, row, left=row[0], right=1.0)


GH5_X, GH5_W = np.polynomial.hermite_e.hermegauss(5)
GH5_W = GH5_W / GH5_W.sum()
GL4_U, GL4_W = np.polynomial.legendre.leggauss(4)
GL4_U, GL4_W = 0.5 * (GL4_U + 1), 0.5 * GL4_W


def hemi_area_ft(q, H):
    'int_0^H (1 - z^2 / H^2) exp(i q z) dz for q (any shape) and heights H (1-d): shape q.shape + (len(H),)'
    q = np.asarray(q, float)[..., None]; H = np.asarray(H, float)
    x = q * H
    u = 1j * np.where(np.abs(x) < 0.05, 1.0, q)
    e = np.exp(u * H)
    closed = -1 / u + 2 * e / (H * u ** 2) - 2 * (e - 1) / (H ** 2 * u ** 3)
    k = np.arange(9)
    from math import factorial
    ck = np.array([(1 / (j + 1) - 1 / (j + 3)) / factorial(j) for j in k])
    series = H * np.sum(ck * (1j * x[..., None]) ** k, axis=-1)
    return np.where(np.abs(x) < 0.05, series, closed)


class Pop:
    def __init__(self, R, a, s, X, W):
        self.R, self.a, self.s = float(R), float(a), float(s)
        self.Rj, self.wj = M.ln_nodes(self.R, self.s, X, W)
        self.Hj = self.a * self.Rj
        self.V = (2 * np.pi / 3) * self.Rj ** 2 * self.Hj
        self.Vm = float(np.sum(self.wj * self.V))

    def area_at(self, z):
        'Number-weighted mean cross-section area at heights z (array) per island.'
        z = np.asarray(z, float)[..., None]
        r2 = np.where(z < self.Hj, self.Rj ** 2 * (1 - (z / self.Hj) ** 2), 0.0)
        return np.sum(self.wj * np.pi * r2, axis=-1)


class Morph2:
    'Small islands (R, a, s, omega; LMA spacing) + large flat clusters (RL, aL, fL); mass thickness h_m.'

    def __init__(self, R, a, s, omega, h_m, RL=20.0, aL=0.3, fL=0.0, kD=1.07, nsize=7):
        self.h_m, self.omega, self.fL = float(h_m), float(omega), float(fL)
        self.S = Pop(R, a, s, GH_X, GH_W) if nsize == 7 else Pop(R, a, s, GH5_X, GH5_W)
        self.L = Pop(RL, aL, 0.3, GH5_X, GH5_W)
        self.nS = (1 - self.fL) * self.h_m / self.S.Vm
        self.nL = self.fL * self.h_m / self.L.Vm
        self.D = kD / np.sqrt(self.nS)
        self.coverage = float(self.nS * np.sum(self.S.wj * np.pi * self.S.Rj ** 2)
                              + self.nL * np.sum(self.L.wj * np.pi * self.L.Rj ** 2))
        self.Hmax = float(max(np.max(self.S.Hj), np.max(self.L.Hj) if self.fL > 0 else 0.0))

        self.eta_hd = float(self.nS * np.sum(self.S.wj * np.pi * (self.S.Rj * (1 + self.omega)) ** 2))

        self.R, self.a, self.s, self.n = self.S.R, self.S.a, self.S.s, self.nS
        self.Rj, self.Hj, self.wj = self.S.Rj, self.S.Hj, self.S.wj

    def phi_profile(self, z_up):
        return self.nS * self.S.area_at(z_up) + (self.nL * self.L.area_at(z_up) if self.fL > 0 else 0.0)

    def slab(self, metal, nslab=8, f1_au=AU_F1_12, fl=False):
        H = self.Hmax
        edges = np.linspace(H, 0, nslab + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        phi = np.clip(self.phi_profile(mids), 0, 1)
        dm = dc(metal, fl=fl, f1_au=f1_au)
        return [(p * dm, H / nslab) for p in phi], phi


def eps_spec(label):
    'IAM_EPS_SPEC="ref15:0.0132,ref24:-0.0038" (2026-11-06, tio2_prereq C2): per-scan eps = alpha_i - 0.40 measured from'
    for item in filter(None, os.environ.get("IAM_EPS_SPEC", "").split(",")):
        k, v = item.split(":")
        if k.strip() == label:
            return float(v)
    return None


class Model2(M.Model):
    def __init__(self, scan, eps=0.0, sig=0.0, nu_isl=0.0, nu_sub=0.0, interf="lma", **kw):
        super().__init__(scan, nu_sub=nu_sub, nu_isl=nu_isl, sig_af=sig, interf=interf, **kw)
        cal = M.tio2_calibration(scan.label) if scan.sub == "TiO2" else None
        es = eps_spec(scan.label)
        if es is not None:
            if cal and abs(float(cal.get("eps") or 0.0)) > 0:
                raise SystemExit(f"{scan.label}: IAM_EPS_SPEC and a calibration eps are both set")
            self._eps_lock = es
            self.set_geom(es)
        else:
            self.set_geom(float(cal["eps"]) if cal and cal.get("eps") is not None else eps)

    _eps_lock = None

    def set_geom(self, eps):
        "alpha_i = alpha_i(intake) + eps (SCANS 'ai', default 0.40); computed terms on alpha_f(true) = alpha_f(nominal) - eps."
        if self._eps_lock is not None and abs(float(eps) - self._eps_lock) > 1e-12:
            raise SystemExit(f"{self.s.label}: eps {eps} requested while IAM_EPS_SPEC fixes it at {self._eps_lock}")
        self.eps = float(eps)
        self.ai = float(self.s.cfg.get("ai", 0.40)) + self.eps
        self.af_c = self.s.af - self.eps
        self.a_all = np.r_[self.ai, np.maximum(self.af_c, 1e-4)]
        self._pre = None

    embed_frac = 0.0
    _e_now = 0.0

    def embed_depth(self, m):
        if self.embed_frac <= 0 or self.s.sub not in M.POLY or not getattr(self, "poly_t", None):
            return 0.0
        return float(min(self.embed_frac * m.S.R * m.S.a, m.Hmax, 0.98 * self.poly_t))

    def embed_slab(self, slab0, phi, m, e):
        'Slices from the top (heights Hmax..0 above the island base); the part below height e is in the polymer.'
        dP = dc(self.s.sub)
        out, top = [], m.Hmax
        for (dd, t), ph in zip(slab0, phi):
            zb = top - t
            f = float(np.clip((e - zb) / t, 0.0, 1.0))
            out.append((dd + f * (1 - ph) * dP, t))
            top = zb
        return out


    def _qz(self):
        return K0 * (np.sin(np.radians(self.ai)) + np.sin(np.radians(np.maximum(self.af_c, 0.0))))

    def r_rep_flat(self, m):
        qz = self._qz()
        rS, rM = self.rho_top, rho_e(self.s.metal, self.f1)
        Fm = 0
        for pop, n in ((m.S, m.nS), (m.L, m.nL)):
            if n <= 0:
                continue
            Fm = Fm + M.ff_sep_sum(self.q, qz, pop.Rj, pop.Hj, n * pop.wj)
        return np.abs(1 + 1j * qz[:, None] * (rM / rS) * Fm) ** 2 - 1

    def islands_sliced(self, m, slab0):
        'Sliced EM DWBA (flat substrate): dsigma/dOmega per area, (n_af, n_q).'
        layers = [(self.nu_isl * d, t) for d, t in slab0]
        a = np.r_[self.ai, np.maximum(self.af_c, 1e-4)]
        kzs, A, B, ts = stack(a, layers + self.flat_layers(), self.d_sub)
        zeta, wz, E = [], [], []
        top = 0.0
        for j in range(1, len(layers) + 1):
            t = ts[j]
            for u, w in zip(GL4_U, GL4_W):
                dz = u * t
                E.append(A[j] * np.exp(1j * kzs[j] * dz) + B[j] * np.exp(-1j * kzs[j] * dz))
                zeta.append(top + dz); wz.append(w * t)
            top += t
        E = np.array(E)
        zeta, wz = np.array(zeta), np.array(wz)
        z_up = m.Hmax - zeta
        if self._e_now > 0:
            c = 1 - rho_e(self.s.sub) / rho_e(self.s.metal, self.f1)
            wz = wz * np.where(z_up < self._e_now, c, 1.0)
        Ei, Ef = E[:, 0], E[:, 1:]
        P = (Ef * (Ei * wz)[:, None]).T
        rho2 = RE ** 2 * abs(rho_e(self.s.metal, self.f1)) ** 2
        out = 0
        Rm = float(np.sum(m.S.wj * m.S.Rj))
        for pop, n, lma in ((m.S, m.nS, True), (m.L, m.nL, False)):
            if n <= 0:
                continue
            S = None
            if lma and self.interf == "hd":
                S = np.array([s_hd(self.q * 2 * Rj * (1 + m.omega), m.eta_hd) for Rj in pop.Rj])
            elif lma:
                S = s_para(self.q[None, :], lma_D(m.D, pop.Rj, Rm)[:, None], m.omega)
            out = out + M.incoherent_sum(P, M.hemi_radius(z_up, pop.Rj, pop.Hj), self.q, n * pop.wj, S)
        return rho2 * out

    def born_pop_tio2(self, pop, n, D, omega, lma, eta=0.0):
        'Orientation-averaged Born of one population on the particle caps (coarse grid -> full grid), per area.'
        if TD.ISLAND:
            return TD.born_pop(self, pop, n, D, omega, lma, eta=eta)
        af, q = self.af_c[self.ia], self.q[self.iq]
        qz = K0 * (np.sin(np.radians(self.ai)) + np.sin(np.radians(np.maximum(af, 0.0))))
        if M.QUAD == "gh":
            pRj, pHj, pwj = pop.Rj, pop.Hj, pop.wj
        else:
            pRj, pwj = M.ln_nodes(pop.R, pop.s, None, None, dr=M.QUAD_DR_ORIENT, rel=M.QUAD_REL_ORIENT)
            pHj = pop.a * pRj
        Rm = float(np.sum(pwj * pRj))
        out = 0
        for th, wt in zip(self.th, self.wth):
            ct, st = np.cos(th), np.sin(th)
            for ph, wp in zip(self.ph, self.wph):
                qzl = q[None, :] * st * np.sin(ph) + qz[:, None] * ct
                qpl = np.sqrt(np.maximum(q[None, :] ** 2 + qz[:, None] ** 2 - qzl ** 2, 0.0))
                acc = 0
                for j0 in range(0, len(pRj), 32):
                    Rj, Hj, wj = pRj[j0:j0 + 32], pHj[j0:j0 + 32], pwj[j0:j0 + 32]
                    F = M.ff_arr_nodes(qpl, qzl, Rj, Hj)
                    I = F.real ** 2 + F.imag ** 2
                    if lma and self.interf == "hd":
                        I = I * np.array([s_hd(qpl * 2 * r_ * (1 + omega), eta * ct) for r_ in Rj])
                    elif lma:
                        I = I * s_para(qpl[None], lma_D(D / np.sqrt(max(ct, 1e-3)), Rj, Rm)[:, None, None], omega)
                    acc = acc + np.einsum("k,kaq->aq", wj, I)
                out = out + wt * wp * acc
        lo = np.log(np.maximum(out, 1e-300))
        A1 = np.array([np.interp(self.af_c, af, lo[:, j]) for j in range(lo.shape[1])]).T
        A2 = np.array([np.interp(np.log(self.q), np.log(q), A1[i]) for i in range(A1.shape[0])])
        return n * np.exp(A2)

    h_shell = None
    cap_frac, cap_phi = 0.0, 1.0
    cap_patch = None
    cap_xi = None
    oz = None
    oz_depth = None
    tfilm = None
    tfilm_x = 1.0

    def r_coat(self, m):
        qz = self._qz()
        qa = np.sqrt(qz[:, None] ** 2 + self.q[None, :] ** 2)
        cpsi = qz[:, None] / qa
        rT, rM = rho_e("TiO2"), rho_e(self.s.metal, self.f1)
        if COAT == "shell" and self.h_shell is not None:
            val = np.abs(1 + (rM / rT) * (np.exp(1j * qa * self.h_shell) - 1) * cpsi) ** 2
            return np.where(np.degrees(np.arccos(np.clip(cpsi, 0, 1))) <= self.theta_max, val, 1.0)
        if TIO2_FIX_A:
            Phi = 0
            for pop, n_ in ((m.S, m.nS), (m.L, m.nL)):
                if n_ > 0:
                    Phi = Phi + n_ * (hemi_area_ft(qa, pop.Hj) @ (pop.wj * np.pi * pop.Rj ** 2))
            Phi = Phi * cpsi
        else:
            zs = np.linspace(0, m.Hmax, 24)
            dz = zs[1] - zs[0]
            Phi = (np.exp(1j * qa[..., None] * zs) @ (m.phi_profile(zs) * dz)) * cpsi
        val = np.abs(1 + 1j * qa * (rM / rT) * Phi) ** 2
        return np.where(np.degrees(np.arccos(np.clip(cpsi, 0, 1))) <= self.theta_max, val, 1.0)

    def tio2_fields(self, h_m, metal=True, k=K0, fl=False):
        'iam.Model.tio2_fields; with IAM_CAP_PLACE=on / IAM_CAP_AVG=int and a cap, also keeps the cap-layer amplitudes'
        I, phm = super().tio2_fields(h_m, metal=metal, k=k, fl=fl)
        if CAP_NEW and metal and self.cap is not None and self.cap[1] > 0:
            dm = dc(self.s.metal, fl=fl, f1_au=self.f1)
            base = self.sub_layers_fl if fl else self.sub_layers
            layers = [(self.nu_sub * self.cap[0] * dm, self.cap[1])]                 + [(d + self.nu_sub * p * dm, t) for (d, t), p in zip(base, phm)]
            a = self.a_all if not fl else np.maximum(getattr(self, "af_c", self.af), 1e-4)
            kzs, A, B, ts = stack(a, layers, self.d_sub_fl if fl else self.d_sub, k=k)
            kz1, t1 = kzs[1], ts[1]
            with np.errstate(over="ignore", invalid="ignore"):
                Bb = np.where(kz1.imag * t1 < 300.0, B[1] * np.exp(-1j * kz1 * t1), 0.0)
            amp = (A[1], Bb, kz1, t1)
            if fl:
                self._cap_amp_fl = amp
            else:
                self._cap_amp, self._Itop = amp, np.abs(A[0] + B[0]) ** 2
        return I, phm

    def tfilm_w(self):
        'IAM_TFILM: film share w(q) = phi_f / (1 + (q L_f / 2 pi)^n) of the low-q scattering.'
        phi_f, L_f = self.tfilm[:2]
        return phi_f / (1.0 + (self.q * L_f / (2 * np.pi)) ** TFILM_N)

    def tfilm_mix(self, w, sub, phm=None):
        "IAM_TFILM: (1 - w(q)) sub + w(q) sub_ff (module comment at TFILM). Returns (sub', sub_ff); sub itself and None"
        phi_f, L_f = self.tfilm[:2]
        own = TFILM_T != "cap"
        self._tfilm_t = None
        if not phi_f > 0 or (self.cap is None and not own):
            return sub, None
        if own:
            t_f = TFILM_KT * float(w["h_m"]) if TFILM_T == "hm" else float(self.tfilm[2])
            if not t_f > 0:
                return sub, None
            dm = dc(self.s.metal, f1_au=self.f1)
            layers = [(self.nu_sub * TFILM_PHI * dm, t_f)] + [(d + self.nu_sub * p * dm, t)
                                                              for (d, t), p in zip(self.sub_layers, phm)]
            _, A_, B_, _ = stack(self.a_all, layers, self.d_sub)
            Itop, phi_m, self._tfilm_t = np.abs(A_[0] + B_[0]) ** 2, TFILM_PHI, t_f
        else:
            Itop, phi_m = self._Itop, self.cap_phi
        wq = self.tfilm_w()
        I0, _ = self.tio2_fields(0.0, metal=False)
        fw = self.tio2_sub_weights()
        top = np.r_[self._region == 0, False]
        if not fw[top].sum() > 0:
            top = np.ones(len(fw), bool)
        Gt = np.sum(fw[top][:, None] * I0[top][:, :1] * I0[top][:, 1:], axis=0) / fw[top].sum()
        ir = np.maximum(np.arange(len(self.af)), int(np.searchsorted(self.af_c, TFILM_AFR)))
        R = Itop[0] * Itop[1:] / np.maximum(Gt[ir], 1e-300)
        if self.Kaf is not None:
            R = self.smear_ratio(R)
        X = abs(phi_m * rho_e(self.s.metal, self.f1) / (self.tio2_fill * rho_e("TiO2"))) ** 2
        if TFILM_X == "fit":
            X = float(self.tfilm_x) * X
        sub_ff = X * w["cfac"] * R[None, :, None] * self.s.I_pre[:, ir, :]
        return (1 - wq)[None, None, :] * sub + wq[None, None, :] * sub_ff, sub_ff

    def _cap_metal_prod(self):
        'Cap share of |E_i|^2 |E_f|^2 for the islands / network: top surface (PLACE=on) or cap average (AVG=int).'
        if CAP_PLACE == "on":
            return self._Itop[0] * self._Itop[1:]
        A, Bb, kz_, t = self._cap_amp
        return cap_avg_product((A[0], Bb[0], kz_[0], t), (A[1:], Bb[1:], kz_[1:], t))

    def predict_tio2(self, w, m, K, g=0.0, f_isl=1.0, parts=False, h_opt=None):
        'TiO2: metal spread over the particle caps for the optics (total mass h_m); a fraction f_isl of the mass'
        rho2 = RE ** 2 * abs(rho_e(self.s.metal, self.f1)) ** 2
        h_opt = m.h_m if h_opt is None else h_opt
        kc = self.cap_frac if self.cap_frac > 0 else 0.0


        c = float(self.cap_patch) if (kc > 0 and self.cap_patch is not None) else 1.0
        self.cap = (self.cap_phi, kc * h_opt / (c * self.cap_phi)) if kc > 0 else None
        I, phm = self.tio2_fields(h_opt * (1 - kc))
        Icap = None
        if self.cap is not None:
            Icap, I = I[0], I[1:]
            if CAP_NEW and (self.cap_xi is not None or self.oz_depth is not None):
                raise SystemExit("IAM_CAP_PLACE=on / IAM_CAP_AVG=int: cap_xi (CAPSCAT) and oz_depth are not supported")
        regions = [(1.0, I)]
        if c < 1.0:
            cap_c, self.cap = self.cap, None
            Io, _ = self.tio2_fields(h_opt * (1 - kc))
            self.cap = cap_c
            regions = [(1.0 - c, Io), (c, I)]
            I = (1.0 - c) * Io + c * I
        fl_w = self.tio2_sub_weights()
        G_all = sum(a * np.sum(fl_w[:, None] * Ir[:, :1] * Ir[:, 1:], axis=0) for a, Ir in regions)
        Gr = G_all / self.pre_factor()
        sub = w["cfac"] * Gr[None, :, None] * self.s.I_pre
        if g > 0 and COAT == "top":
            if M.TIO2_NUM:
                rp = getattr(self, "_rho_peak", 0.0)
                u = np.r_[phm / max(rp, 1e-12), 0.0] if rp > 0 else np.zeros(len(fl_w))
            else:
                u = np.r_[phm / max(phm.max(), 1e-12), 0.0] if phm.max() > 0 else np.zeros(len(fl_w))
            T = sum(a * np.sum((fl_w * u)[:, None] * Ir[:, :1] * Ir[:, 1:], axis=0) for a, Ir in regions) \
                / np.maximum(G_all, 1e-300)
            sub = sub * (1 + g * T[:, None] * (self.r_coat(m) - 1))[None]
        elif g > 0:
            sub = sub * (1 - g + g * self.r_coat(m))[None]
        if M.TIO2_NUM:
            tl = np.array([tt for _, tt in self.sub_layers])[: len(phm)]
            wt = phm * tl / max((phm * tl).sum(), 1e-12)
        else:
            wt = phm / max(phm.sum(), 1e-12)
        wm = wt
        if TD.DWBA != 0 or TD.NET != 0 or TD.CONTRAST != "vac":
            if self.cap is not None or c < 1.0:
                raise SystemExit("tio2_dwba switches: cap layers / patches are not supported")
            wt = wt * TD.contrast_weights(self, phm, self.fills, rho_e("TiO2"), rho_e(self.s.metal, self.f1))
            if TD.DWBA != 0 or TD.NET != 0:
                self._amp = TD.amplitudes(self, phm, wt, wm)
        Is = I[: len(phm)]
        Ef2 = top2 = None
        Ei, Ef = float(np.sum(wt * Is[:, 0])), np.sum(wt[:, None] * Is[:, 1:], axis=0)
        if Icap is not None:
            Ei, Ef = kc * float(Icap[0]) + (1 - kc) * Ei, kc * Icap[1:] + (1 - kc) * Ef
        if TIO2_FIX_B:
            def avgprod(wz):
                return sum(a * np.sum((wz[:, None] * Ir[: len(wz), :1]) * Ir[: len(wz), 1:], axis=0) for a, Ir in regions)
            EiEf = avgprod(wt)
            if Icap is not None and not CAP_NEW:
                EiEf = kc * Icap[0] * Icap[1:] + (1 - kc) * EiEf
            elif Icap is not None:
                kcP = kc * self._cap_metal_prod()
                EiEf = kcP + (1 - kc) * EiEf
                top_share = kcP / np.maximum(EiEf, 1e-300)
                if TFILM_MET and self.tfilm is not None and self.tfilm[0] > 0:
                    Ptop, s_q = self._cap_metal_prod(), kc + (1 - kc) * self.tfilm_w()
                    Ef2 = s_q[None, :] * Ptop[:, None] + (1 - s_q)[None, :] * avgprod(wt)[:, None]
                    top2 = s_q[None, :] * Ptop[:, None] / np.maximum(Ef2, 1e-300)
            Ei, Ef = 1.0, EiEf
        self._EiEf = Ei * Ef
        born = self.born_pop_tio2(m.S, m.nS, m.D, m.omega, True, eta=m.eta_hd)
        if m.fL > 0:
            born = born + self.born_pop_tio2(m.L, m.nL, 1.0, 0.0, False)
        isl = f_isl * Ei * (Ef[:, None] if Ef2 is None else Ef2) * rho2 * born
        if self.oz is not None:
            A, xi, H = self.oz[:3]
            vz = np.exp(-(self._qz() * self.oz[3]) ** 2) if len(self.oz) > 3 else 1.0
            En_i, En_f = Ei, Ef
            if self.oz_depth is not None:
                from scipy.special import erf
                t_s = np.array([tt for _, tt in self.sub_layers]); zb = np.r_[0.0, np.cumsum(t_s)]
                zn, wn = self.oz_depth
                wz = np.diff(0.5 * (1 + erf((zb - zn) / (np.sqrt(2) * max(wn, 1e-3)))))
                wz = wz / max(wz.sum(), 1e-12)
                En_i, En_f = float(np.sum(wz * Is[:, 0])), np.sum(wz[:, None] * Is[:, 1:], axis=0)
                if TIO2_FIX_B:
                    En_i, En_f = 1.0, avgprod(wz[: len(phm)] if len(wz) >= len(phm) else np.r_[wz, np.zeros(len(phm) - len(wz))])
            if TD.NET != 0:
                if self.oz_depth is not None or not TIO2_FIX_B:
                    raise SystemExit("IAM_TIO2_DWBA_NET needs IAM_TIO2_FIX=1 and the network at the metal depth")
                sz_ = self.oz[3] if len(self.oz) > 3 else 0.0
                vzf = TD.network_vertical(self, sz_) if TD.NET == 1 else TD.network_double_integral(self, sz_)
                self._net_V = vzf
            else:
                vzf = En_f * vz
            if Ef2 is not None:
                vzf = Ef2 * (vz[:, None] if np.ndim(vz) else vz)
            isl = isl + (A * 8 * np.pi * h_opt * xi ** 3 * En_i * rho2) * (vzf[:, None] if Ef2 is None else vzf) \
                * (1.0 / (1 + (self.q * xi) ** 2) ** (1 + H))[None, :]
        if self.cap_xi is not None and Icap is not None and c < 1.0:


            phi_c, t_c = self.cap
            qz = self._qz()
            vert = np.where(qz * t_c > 1e-6, (2 * np.sin(0.5 * qz * t_c) / np.maximum(qz, 1e-12)) ** 2, t_c ** 2)
            lat = c * (1.0 - c) * 2 * np.pi * self.cap_xi ** 2 / (1 + (self.q * self.cap_xi) ** 2) ** 1.5
            isl = isl + (RE ** 2 * abs(phi_c * rho_e(self.s.metal, self.f1)) ** 2) \
                * (Icap[0] * Icap[1:] * vert)[:, None] * lat[None, :]
        fl = np.zeros(len(self.af))
        if self.s.metal == "Au" and m.h_m > 0:
            If, _ = self.tio2_fields(h_opt * (1 - kc), k=K_FL, fl=True)
            if self.cap is not None:
                Ifb = If[1: 1 + len(phm)]
                if c < 1.0:
                    cap_c, self.cap = self.cap, None
                    Ifo, _ = self.tio2_fields(h_opt * (1 - kc), k=K_FL, fl=True)
                    self.cap = cap_c
                    Ifb = (1.0 - c) * Ifo[: len(phm)] + c * Ifb
                Eff = kc * If[0] + (1 - kc) * np.sum(wm[:, None] * Ifb, axis=0)
            else:
                Eff = np.sum(wm[:, None] * If[: len(phm)], axis=0)
            Ei_m = Ei if wm is wt else float(np.sum(wm * Is[:, 0]))
            fl = self.eps_fl * EFF_RATIO * P_L3 * MU_AU_12 * h_opt * Ei_m * OMEGA_L3 / (4 * np.pi) * Eff
            if TIO2_FIX_B:
                n_ = len(phm)
                if self.cap is not None:
                    if c < 1.0:
                        fl_reg = [(1.0 - c, regions[0][1], Ifo[:n_]), (c, regions[1][1], If[1: 1 + n_])]
                    else:
                        fl_reg = [(1.0, regions[0][1], If[1: 1 + n_])]
                    if CAP_AVG == "int":
                        A, Bb, kz_, t = self._cap_amp
                        capfl = kc * cap_avg_product((A[0], Bb[0], kz_[0], t), self._cap_amp_fl)
                    else:
                        capfl = kc * Icap[0] * If[0]
                    EiEff = capfl + (1 - kc) * sum(
                        a * np.sum((wm[:, None] * Ir[:n_, :1]) * Fr, axis=0) for a, Ir, Fr in fl_reg)
                else:
                    EiEff = sum(a * np.sum((wm[:, None] * Ir[:n_, :1]) * If[:n_], axis=0) for a, Ir in regions)
                fl = self.eps_fl * EFF_RATIO * P_L3 * MU_AU_12 * h_opt * OMEGA_L3 / (4 * np.pi) * EiEff
        metal = self.smear(isl + fl[:, None])
        if self.Kaf is not None:
            sub = sub * (self.smear_ratio(Gr) / np.maximum(Gr, 1e-12))[None, :, None]
        sub_ff = None
        if self.tfilm is not None:
            sub, sub_ff = self.tfilm_mix(w, sub, phm)
        mu = M.shadow_add(w, sub + K * metal[None])
        if parts:
            pr = dict(sub=sub, metal=metal, isl=isl, fl=fl, G=Gr, coverage=m.coverage, D=m.D)
            if sub_ff is not None:
                pr["sub_ff"] = sub_ff
            if CAP_PLACE == "on" and Icap is not None:
                pr["isl_top"] = (top_share[:, None] if top2 is None else top2) * isl
            return mu, pr
        return mu, sub, metal

    def predict(self, w, m, K, g=0.0, parts=False, f_isl=1.0):
        if self.s.sub == "TiO2":
            return self.predict_tio2(w, m, K, g=g, f_isl=f_isl, parts=parts)
        pre = self.pre_factor()
        e = self.embed_depth(m)
        if e > 0:
            saved = self.sub_layers, self.sub_layers_fl
            self.sub_layers = [(dc(self.s.sub), self.poly_t - e)]
            self.sub_layers_fl = [(dc(self.s.sub, fl=True), self.poly_t - e)]
            self._e_now = e
            try:
                return self._predict_flat(w, m, K, g, parts, pre, e)
            finally:
                self.sub_layers, self.sub_layers_fl = saved
                self._e_now = 0.0
        return self._predict_flat(w, m, K, g, parts, pre, 0.0)

    def _predict_flat(self, w, m, K, g, parts, pre, e):
        slab0, phi = m.slab(self.s.metal, f1_au=self.f1)
        slab = [(self.nu_sub * d, t) for d, t in slab0]
        if e > 0:
            slab0 = self.embed_slab(slab0, phi, m, e)
            slab = slab0
        Ise, G = self.fields(slab)
        Gr = G / pre
        sub = w["cfac"] * Gr[None, :, None] * self.s.I_pre
        if g > 0:
            sub = sub * (1 + g * self.r_rep_flat(m))[None]
        isl = self.islands_sliced(m, slab0)
        if self.oz is not None and len(slab):
            A, xi, H = self.oz[:3]
            vz = np.exp(-(self._qz() * self.oz[3]) ** 2) if len(self.oz) > 3 else 1.0
            rho2 = RE ** 2 * abs(rho_e(self.s.metal, self.f1)) ** 2
            isl = isl + (A * 8 * np.pi * m.h_m * xi ** 3 * float(Ise[-1, 0]) * rho2) * (Ise[-1, 1:] * vz)[:, None] \
                * (1.0 / (1 + (self.q * xi) ** 2) ** (1 + H))[None, :]
        fl = self.fluor(m, slab, phi, Ise)
        metal = self.smear(isl + fl[:, None])
        if self.Kaf is not None:
            sub = sub * (self.smear_ratio(Gr) / np.maximum(Gr, 1e-12))[None, :, None]
        mu = M.shadow_add(w, sub + K * metal[None])
        if parts:
            return mu, dict(sub=sub, metal=metal, isl=isl, fl=fl, G=Gr, coverage=m.coverage, D=m.D)
        return mu, sub, metal


__all__ = ["Morph2", "Model2"]
