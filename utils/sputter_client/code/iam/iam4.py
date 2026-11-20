'IAM v0.4: densely packed columnar grains (one description from islands to a granular film).'

from __future__ import annotations

import os

import numpy as np

import iam as M
import iam2 as M2
from iam import K0, RE, dc, rho_e, stack, jinc, s_para, lma_D, GH_X, GH_W

GL4_U, GL4_W = M2.GL4_U, M2.GL4_W


class Grains:
    def __init__(self, R, s, eta, fcap, omega, h_m, RL=20.0, aL=0.25, fL=0.0):
        self.R, self.s, self.eta, self.fcap, self.omega = float(R), float(s), float(eta), float(fcap), float(omega)
        self.h_m, self.fL = float(h_m), float(fL)
        self.Rj, self.wj = M.ln_nodes(self.R, self.s, GH_X, GH_W)
        self.Rm = float(np.sum(self.wj * self.Rj))
        self.n = self.eta / float(np.sum(self.wj * np.pi * self.Rj ** 2))
        self.T = (1 - self.fL) * self.h_m / (self.eta * (1 - self.fcap / 3))
        self.Hc, self.Hcap = (1 - self.fcap) * self.T, self.fcap * self.T
        self.D = self.Rm * np.sqrt(2 * np.pi / (np.sqrt(3) * self.eta))
        self.L = M2.Pop(RL, aL, 0.3, M2.GH5_X, M2.GH5_W)
        self.nL = self.fL * self.h_m / self.L.Vm if self.fL > 0 else 0.0
        self.Hmax = float(max(self.T, np.max(self.L.Hj) if self.fL > 0 else 0.0))
        self.coverage = self.eta + (self.nL * np.sum(self.L.wj * np.pi * self.L.Rj ** 2) if self.fL > 0 else 0.0)


    sH = 0.0


    kD = 1.0

    def height_nodes(self):
        '(weights, Hc, Hcap) of the grain-height nodes: T exp(sH x - sH^2 / 2) on 5 Gauss-Hermite nodes.'
        if self.sH <= 0:
            return np.array([1.0]), np.array([self.Hc]), np.array([self.Hcap])
        f = np.exp(self.sH * M2.GH5_X - 0.5 * self.sH ** 2)
        return M2.GH5_W, self.Hc * f, self.Hcap * f

    def set_height_spread(self, sH):
        self.sH = float(sH)
        if self.sH > 0:
            _, Hc, Hcap = self.height_nodes()
            self.Hmax = float(max(self.Hmax, np.max(Hc + Hcap)))

    def rel_radius_h(self, z, Hc, Hcap):
        'r(z) / R for a grain with column height Hc and cap height Hcap.'
        z = np.asarray(z, float)
        out = np.where(z < Hc, 1.0, 0.0)
        if Hcap > 0:
            u = (z - Hc) / Hcap
            out = np.where((z >= Hc) & (u < 1), np.sqrt(np.clip(1 - u * u, 0, None)), out)
        return np.where(z < 0, 0.0, out)

    def rel_radius(self, z):
        'r(z) / R for a grain (same shape for all sizes; the mean height with sH > 0).'
        return self.rel_radius_h(z, self.Hc, self.Hcap)

    def rel_radius_nodes(self, z):
        '[(weight, r(z) / R)] over the height nodes.'
        w, Hc, Hcap = self.height_nodes()
        if self.sH <= 0:
            return [(1.0, self.rel_radius(z))]
        return [(float(wk), self.rel_radius_h(z, a, b)) for wk, a, b in zip(w, Hc, Hcap)]

    def phi_profile(self, z_up):
        z = np.asarray(z_up, float)
        g = self.eta * self.rel_radius(z) ** 2 if self.sH <= 0 else             self.eta * sum(wk * rr ** 2 for wk, rr in self.rel_radius_nodes(z))
        if self.fL > 0:
            g = g + self.nL * self.L.area_at(z)
        return g

    def slab(self, metal, nslab=None, f1_au=M.AU_F1_12):
        'EM slices of <= 1 nm (the depth integral of the field product is done with 4 nodes per slice).'
        H = self.Hmax
        if nslab is None:
            nslab = M.SMOOTH_NSLAB if M.SMOOTH else max(8, int(np.ceil(H / 1.0)))
        edges = np.linspace(H, 0, nslab + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        phi = np.clip(self.phi_profile(mids), 0, 1)
        dm = dc(metal, f1_au=f1_au)
        return [(p * dm, H / nslab) for p in phi], phi


class Model4(M2.Model2):


    sL = 0.0


    sigL = None
    KafL = None
    _islL = None

    def set_smear_L(self, sig):
        self.sigL = float(sig)
        self.KafL = self.smear_kernel(self.sigL)

    def smear_L(self, A):
        return A if self.KafL is None else self.KafL @ A

    def smear_metal(self, isl, fl):
        'Smeared metal term of predict_grains: isl (the grains_sliced total, plus terms added to it) + fluorescence'
        L = self._islL if self.sigL is not None else None
        if L is None:
            return self.smear(isl + fl[:, None]), None
        return self.smear(isl - L + fl[:, None]) + self.smear_L(L), L

    def _nodes(self, m, slab0):
        layers = [(self.nu_isl * d, t) for d, t in slab0]
        a = np.r_[self.ai, np.maximum(self.af_c, 1e-4)]
        kzs, A, B, ts = stack(a, layers, self.d_sub)
        E, zeta, wz = [], [], []
        split = ([], [], [])
        top = 0.0
        for j in range(1, len(layers) + 1):
            for u, w in zip(GL4_U, GL4_W):
                dz = u * ts[j]
                if self.sL > 0:
                    ea, eb = A[j] * np.exp(1j * kzs[j] * dz), B[j] * np.exp(-1j * kzs[j] * dz)
                    E.append(ea + eb); split[0].append(ea); split[1].append(eb); split[2].append(kzs[j])
                else:
                    E.append(A[j] * np.exp(1j * kzs[j] * dz) + B[j] * np.exp(-1j * kzs[j] * dz))
                zeta.append(top + dz); wz.append(w * ts[j])
            top += ts[j]
        E = np.array(E); z_up = m.Hmax - np.array(zeta); wz = np.array(wz)
        self._split = split
        return E, z_up, wz

    def _p_soft(self, wz):
        'Field product of the last _nodes call for a vertical profile smoothed by a Gaussian of rms sL (nm). In a'
        Ea, Eb, K = (np.array(v) for v in self._split)
        s2 = 0.5 * self.sL ** 2
        ki, kf = K[:, :1], K[:, 1:]
        P = 0.0
        for ei, si in ((Ea[:, :1], 1.0), (Eb[:, :1], -1.0)):
            for ef, sf in ((Ea[:, 1:], 1.0), (Eb[:, 1:], -1.0)):
                P = P + ei * ef * np.exp(-s2 * (si * ki + sf * kf) ** 2)
        return (P * wz[:, None]).T

    def grains_sliced(self, m: Grains, slab0):
        E, z_up, wz = self._nodes(m, slab0)
        P = (E[:, 1:] * (E[:, 0] * wz)[:, None]).T
        rho2 = RE ** 2 * abs(rho_e(self.s.metal, self.f1)) ** 2
        S = M.s_lat(self.q[None, :], lma_D(m.D * getattr(m, "kD", 1.0), m.Rj, m.Rm)[:, None], m.omega)
        out = 0.0
        for wk, rr in m.rel_radius_nodes(z_up):

            cyl = rr >= 1.0
            Pe = np.c_[P[:, cyl].sum(1), P[:, ~cyl]]
            rre = np.r_[1.0, rr[~cyl]]
            out = out + wk * M.incoherent_sum(Pe, m.Rj[:, None] * rre[None, :], self.q, m.n * m.wj, S)
        self._islL = self._accL = None
        if m.fL > 0:
            PL = self._p_soft(wz) if self.sL > 0 else P
            accL = M.incoherent_sum(PL, M.hemi_radius(z_up, m.L.Rj, m.L.Hj), self.q, m.nL * m.L.wj)
            out = out + accL
            self._accL = rho2 * accL
            if self.sigL is not None:
                self._islL = self._accL
        return rho2 * out

    def r_rep_grains(self, m: Grains, nz=None):
        '|1 + i q_z (rho_M / rho_Si) n <F(q_par, q_z)>|^2 - 1 with the grain amplitude (Born, vacuum q_z).'
        qz = self._qz()
        rS, rM = rho_e("Si"), rho_e(self.s.metal, self.f1)
        if nz is None:
            nz = M.SMOOTH_NZ if M.SMOOTH else max(48, int(np.ceil(m.Hmax / 0.25)))
        z = (np.arange(nz) + 0.5) / nz * m.Hmax; dz = m.Hmax / nz
        ph = np.exp(1j * qz[:, None] * z[None, :]) * dz
        Fm = 0.0
        for wk, rr in m.rel_radius_nodes(z):
            cyl = rr >= 1.0
            phe = np.c_[ph[:, cyl].sum(1), ph[:, ~cyl]]
            rre = np.r_[1.0, rr[~cyl]]
            Fm = Fm + wk * (phe @ M.coherent_sig(m.Rj[:, None] * rre[None, :], self.q, m.n * m.wj))
        if m.fL > 0:
            FL = ph @ M.coherent_sig(M.hemi_radius(z, m.L.Rj, m.L.Hj), self.q, m.nL * m.L.wj)
            if self.sL > 0:
                FL = FL * np.exp(-0.5 * (qz * self.sL) ** 2)[:, None]
            Fm = Fm + FL
        return np.abs(1 + 1j * qz[:, None] * (rM / rS) * Fm) ** 2 - 1

    def predict_grains(self, w, m: Grains, K, g=0.0, parts=False):
        if self.s.sub == "TiO2":
            raise NotImplementedError("grain model on TiO2: next step")
        slab0, phi = m.slab(self.s.metal, f1_au=self.f1)
        slab = [(self.nu_sub * d, t) for d, t in slab0]
        Ise, G = self.fields(slab)
        Gr = G / self.pre_factor()
        sub = w["cfac"] * Gr[None, :, None] * self.s.I_pre
        if g > 0:
            sub = sub * (1 + g * self.r_rep_grains(m))[None]
        isl = self.grains_sliced(m, slab0)
        fl = self.fluor(M.Morph(1.0, 1.0, 0.1, 0.3, m.h_m), slab, phi, Ise) if m.h_m > 0 else np.zeros(len(self.af))
        metal, L = self.smear_metal(isl, fl)
        if self.Kaf is not None:
            sub = sub * (self.smear_ratio(Gr) / np.maximum(Gr, 1e-12))[None, :, None]
        mu = M.shadow_add(w, sub + K * metal[None])
        if parts:
            return mu, dict(sub=sub, isl=isl, fl=fl, G=Gr, T=m.T, D=m.D, **({} if L is None else {"isl_L": L}))
        return mu, sub, metal


__all__ = ["Grains", "Model4"]


class Grains2(Grains):
    'Columnar grains + small hemispheroidal islands (R_s, a_s, lognormal 0.25) in the free area 1 - eta with local'

    def __init__(self, R, s, eta, fcap, omega, h_m, RL=20.0, aL=0.25, fL=0.0, Rs=1.0, as_=1.0, cs=0.0, oms=0.35):
        self.sp = M2.Pop(Rs, as_, 0.25, M2.GH5_X, M2.GH5_W)
        gap = max(1 - eta, 0.02)
        n_s = cs * gap / float(np.sum(self.sp.wj * np.pi * self.sp.Rj ** 2))
        fs = min(n_s * self.sp.Vm / max(h_m, 1e-9), 0.9 - fL)
        super().__init__(R, s, eta, fcap, omega, h_m * (1 - fs) / max(1 - fs, 1e-9), RL=RL, aL=aL, fL=fL)
        self.fs, self.oms = fs, oms
        self.n_s = fs * h_m / self.sp.Vm
        self.D_s = 1.07 / np.sqrt(max(self.n_s / gap, 1e-12))
        self.T = (1 - fL - fs) * h_m / (self.eta * (1 - self.fcap / 3))
        self.Hc, self.Hcap = (1 - self.fcap) * self.T, self.fcap * self.T
        self.Hs = float(np.max(self.sp.Hj)) if fs > 0 else 0.0
        self.Hmax = float(max(self.T, self.Hs, np.max(self.L.Hj) if self.fL > 0 else 0.0))

    def phi_profile(self, z_up):
        g = super().phi_profile(z_up)
        if self.fs > 0:
            g = g + self.n_s * self.sp.area_at(np.asarray(z_up, float))
        return g


class Model4b(Model4):
    def grains_sliced(self, m, slab0):
        out = super().grains_sliced(m, slab0)
        if getattr(m, "fs", 0) > 0:
            E, z_up, wz = self._nodes(m, slab0)
            P = (E[:, 1:] * (E[:, 0] * wz)[:, None]).T
            Rm = float(np.sum(m.sp.wj * m.sp.Rj))
            S = M.s_lat(self.q[None, :], lma_D(m.D_s, m.sp.Rj, Rm)[:, None], m.oms)
            acc = M.incoherent_sum(P, M.hemi_radius(z_up, m.sp.Rj, m.sp.Hj), self.q, m.n_s * m.sp.wj, S)
            out = out + RE ** 2 * abs(rho_e(self.s.metal, self.f1)) ** 2 * acc
        return out


class GrainsB(Grains2):
    'Grains2 on a laterally homogeneous base layer (the coalesced film; its grain-boundary voids are not resolved).'


    fill_d = None

    def __init__(self, R, s, eta, fcap, omega, h_m, fb=0.0, rb=1.0, **kw):
        self.fb, self.rb, self.h_tot = float(fb), float(rb), float(h_m)
        self.t_b = self.fb * self.h_tot / self.rb
        super().__init__(R, s, eta, fcap, omega, (1 - self.fb) * self.h_tot, **kw)
        self.n_top = None

    def slab(self, metal, nslab=None, f1_au=M.AU_F1_12):
        'Grain slices (top) + base slices of <= 2 nm (bottom); n_top = number of grain slices.'
        top, phi = super().slab(metal, nslab, f1_au)
        self.n_top = len(top)
        if self.t_b <= 1e-6:
            return top, phi
        nb = M.SMOOTH_NB if M.SMOOTH else max(1, int(np.ceil(self.t_b / 2.0)))
        dm = dc(metal, f1_au=f1_au)
        db = self.rb * dm if self.fill_d is None else self.rb * dm + (1.0 - self.rb) * self.fill_d
        return top + [(db, self.t_b / nb)] * nb, np.r_[phi, np.full(nb, self.rb)]


class Model4c(Model4b):
    'Model4b with one optics for every term (the nu-scaled metal slices, base included, set the scatterer fields, the'

    bg = 1.0
    rough = None


    pfilm = os.environ.get("IAM_PFILM", "0") == "1"


    void = None
    buried = None
    host = None

    def _host(self):
        'Layers between the base and the substrate without buried metal (the pre-state reference).'
        if self.host is not None:
            return list(self.host)
        return list(self.flat_layers()) if self.pfilm else []

    _fill_dt = 0.0

    def _below(self):
        if self.buried is None:
            host = self._host()
            if self._fill_dt > 0 and host:
                (d0, t0), rest = host[0], host[1:]
                host = [(d0, max(t0 - self._fill_dt, 0.0))] + list(rest)
            return host
        if self._fill_dt > 0:
            raise NotImplementedError("IAM_PFILL with IAM_BURIED")
        import underlayer as U
        return U.buried_layers(self, self._host())

    def _fill_rho(self, m):
        'Electron density of the polymer filling the base (0 without IAM_PFILL).'
        return self.rho_top if getattr(m, "fill_d", None) is not None else 0.0

    def _sub_rho(self, below):
        return self.rho_top if below else rho_e("Si")

    def base_rough(self, m, slab):
        'Diffuse scattering of the rough base top surface (DWBA, small roughness; Sinha / k-correlation form):'
        nt = m.n_top if getattr(m, "n_top", None) is not None else len(slab)
        if self.rough is None or len(slab) <= nt:
            return 0.0
        kzs, A, B, ts = stack(self.a_all, slab + self._below(), self.d_sub)
        Et = A[nt + 1] + B[nt + 1]
        P = np.abs(Et[0] * Et[1:]) ** 2
        dd = slab[nt][0] - (slab[nt - 1][0] if nt > 0 else 0.0)
        Ar, xi, H = self.rough
        C = Ar * xi ** 2 / (1 + (self.q * xi) ** 2) ** (1 + H)
        return RE ** 2 * abs(dd / M.pref(M.LAM)) ** 2 * P[:, None] * C[None, :]

    def _nodes(self, m, slab0):
        nt = m.n_top if getattr(m, "n_top", None) is not None else len(slab0)
        layers = [(self.nu_isl * d, t) for d, t in slab0]
        a = np.r_[self.ai, np.maximum(self.af_c, 1e-4)]
        kzs, A, B, ts = stack(a, layers + self._below(), self.d_sub)
        E, zeta, wz = [], [], []
        split = ([], [], [])
        top = 0.0
        for j in range(1, nt + 1):
            for u, w in zip(GL4_U, GL4_W):
                dz = u * ts[j]
                if self.sL > 0:
                    ea, eb = A[j] * np.exp(1j * kzs[j] * dz), B[j] * np.exp(-1j * kzs[j] * dz)
                    E.append(ea + eb); split[0].append(ea); split[1].append(eb); split[2].append(kzs[j])
                else:
                    E.append(A[j] * np.exp(1j * kzs[j] * dz) + B[j] * np.exp(-1j * kzs[j] * dz))
                zeta.append(top + dz); wz.append(w * ts[j])
            top += ts[j]
        E = np.array(E); z_up = m.Hmax - np.array(zeta); wz = np.array(wz)
        self._split = split
        return E, z_up, wz

    def substrate_ratio(self, m, slab):
        if self.host is not None:
            raise NotImplementedError("substrate term under an explicit host: computed by the host's own model")
        nt = m.n_top if getattr(m, "n_top", None) is not None else len(slab)
        below = self._below()
        kzs, A, B, ts = stack(self.a_all, slab + below, self.d_sub)
        _, Ab, Bb, _ = stack(self.a_all, self._host(), self.d_sub)
        Eb, Ebare = A[len(slab) + 1] + B[len(slab) + 1], Ab[1] + Bb[1]
        P_bot, P_bare = Eb[0] * Eb[1:], Ebare[0] * Ebare[1:]
        rS = self._sub_rho(below)
        if len(slab) > nt:
            Et = A[nt + 1] + B[nt + 1]
            P_top, r_b = Et[0] * Et[1:], self.nu_isl * m.rb * rho_e(self.s.metal, self.f1) + (1.0 - m.rb) * self._fill_rho(m)
        else:
            P_top, r_b = P_bot, 0.0
        return np.abs(rS * P_bot + r_b * (P_top - P_bot)) ** 2 / np.abs(rS * P_bare) ** 2


    fluct = None
    fluct_s = 0.3


    fluct_psd = False
    fluct_h = 0.5


    fluct_d = None
    _ff = None


    dom = None
    _dom = None

    def domain_term(self, m, slab0):
        'dsigma/dOmega (n_af, n_q) of the metal-fraction modulation inside the composite base, or None when off.'
        if self.dom is None or not self.dom[2] > 2e-6:
            return None
        nt = m.n_top if getattr(m, "n_top", None) is not None else len(slab0)
        if len(slab0) <= nt or getattr(m, "t_b", 0.0) <= 1e-6:
            return np.zeros((len(self.af_c), len(self.q)))
        Ld, xd, Ad = (float(v) for v in self.dom)
        layers = [(self.nu_isl * d, t) for d, t in slab0]
        a = np.r_[self.ai, np.maximum(self.af_c, 1e-4)]
        kzs, A, B, ts = stack(a, layers + self._below(), self.d_sub)
        rM = rho_e(self.s.metal, self.f1)
        c = rM - self._fill_rho(m)
        amp = 0.0
        for j in range(nt + 1, len(layers) + 1):
            for u, w in zip(GL4_U, GL4_W):
                dz = u * ts[j]
                E = A[j] * np.exp(1j * kzs[j] * dz) + B[j] * np.exp(-1j * kzs[j] * dz)
                amp = amp + m.rb * w * ts[j] * E[0] * E[1:]
        k2, xi2 = (2.0 * np.pi / Ld) ** 2, xd ** -2.0
        C = np.pi * xd ** 2 * (k2 + xi2) ** 2 / ((self.q ** 2 - k2 + xi2) ** 2 + 4.0 * k2 * xi2)
        return RE ** 2 * abs(c) ** 2 * Ad * (np.abs(amp) ** 2)[:, None] * C[None, :]

    def film_fluct(self, m, slab0):
        'dsigma/dOmega per area of the film-density fluctuation, (n_af, n_q), or None when off:'
        if self.fluct is None or not self.fluct[1] > 2e-6:
            return None
        RP, AP = self.fluct
        layers = [(self.nu_isl * d, t) for d, t in slab0]
        a = np.r_[self.ai, np.maximum(self.af_c, 1e-4)]
        kzs, A, B, ts = stack(a, layers + self._below(), self.d_sub)
        dm = dc(self.s.metal, f1_au=self.f1)
        nt = m.n_top if getattr(m, "n_top", None) is not None else len(slab0)
        filled = getattr(m, "fill_d", None) is not None
        cbase = (1.0 - self._fill_rho(m) / rho_e(self.s.metal, self.f1)) if filled else 1.0
        amp = 0.0
        for j in range(1, len(layers) + 1):
            if filled and j > nt:
                phi = m.rb * cbase
            else:
                phi = float(np.real(slab0[j - 1][0] / dm))
            for u, w in zip(GL4_U, GL4_W):
                dz = u * ts[j]
                E = A[j] * np.exp(1j * kzs[j] * dz) + B[j] * np.exp(-1j * kzs[j] * dz)
                amp = amp + phi * w * ts[j] * E[0] * E[1:]
        if self.fluct_d is not None:
            k2, xi2 = (2.0 * np.pi / self.fluct_d) ** 2, RP ** -2.0
            C = np.pi * RP ** 2 * (k2 + xi2) ** 2 / ((self.q ** 2 - k2 + xi2) ** 2 + 4.0 * k2 * xi2)
            return RE ** 2 * abs(rho_e(self.s.metal, self.f1)) ** 2 * AP * (np.abs(amp) ** 2)[:, None] * C[None, :]
        if self.fluct_psd:
            C = np.pi * RP ** 2 * (1.0 + (self.q * RP) ** 2) ** -(1.0 + self.fluct_h)
            return RE ** 2 * abs(rho_e(self.s.metal, self.f1)) ** 2 * AP * (np.abs(amp) ** 2)[:, None] * C[None, :]
        Rj, wj = M.ln_nodes(RP, self.fluct_s, M2.GH5_X, M2.GH5_W)
        nP = AP / (np.pi * np.sum(wj * Rj ** 2))
        return RE ** 2 * abs(rho_e(self.s.metal, self.f1)) ** 2 * M.incoherent_sum(amp[:, None], Rj[:, None], self.q, nP * wj)

    def grains_sliced(self, m, slab0):
        out = super().grains_sliced(m, slab0)
        self._void_part = self._bur_part = None
        if self.void is not None or self.buried is not None:
            import underlayer as U
            self._void_part, self._bur_part = U.void_term(self, m, slab0), U.buried_term(self, m, slab0)
            out = out + self._void_part + self._bur_part
        self._dom = dt = self.domain_term(m, slab0)
        if dt is not None:
            out = out + dt
        self._ff = ff = self.film_fluct(m, slab0)
        if ff is None:
            return out
        if self.sigL is not None:
            self._islL = ff if self._islL is None else self._islL + ff
        return out + ff

    def predict_grains(self, w, m, K, g=0.0, parts=False):
        if self.s.sub == "TiO2":
            raise NotImplementedError("base-layer model on TiO2")
        self.nu_sub = self.nu_isl
        if getattr(m, "fill_d", None) is not None:
            if abs(self.nu_isl - 1.0) > 1e-12:
                raise NotImplementedError("IAM_PFILL needs nu = 1")
            self._fill_dt = (1.0 - m.rb) * getattr(m, "t_b", 0.0)
        else:
            self._fill_dt = 0.0
        slab0, phi = m.slab(self.s.metal, f1_au=self.f1)
        slab = [(self.nu_isl * d, t) for d, t in slab0]
        ratio = self.bg + (1 - self.bg) * self.substrate_ratio(m, slab)
        sub = w["cfac"] * ratio[None, :, None] * self.s.I_pre
        if g > 0:
            sub = sub * (1 + g * self.r_rep_grains(m))[None]
        isl = self.grains_sliced(m, slab0) + self.base_rough(m, slab)
        Ise = M.layer_mid_int(self.a_all, slab + self.sub_layers, self.d_sub)[: len(slab)]
        h_tot = getattr(m, "h_tot", m.h_m)
        if self.buried is not None and self.s.metal in ("Au", "Cu"):
            import underlayer as U
            fl = U.buried_fluor(self, slab, phi, Ise, h_tot)
        else:
            fl = self.fluor(M.Morph(1.0, 1.0, 0.1, 0.3, h_tot), slab, phi, Ise) if h_tot > 0 else np.zeros(len(self.af))
        metal, L = self.smear_metal(isl, fl)
        if self.Kaf is not None:
            sub = sub * (self.smear_ratio(ratio) / np.maximum(ratio, 1e-12))[None, :, None]
        mu = M.shadow_add(w, sub + K * metal[None], self)
        if parts:
            extra = {}
            if self.void is not None or self.buried is not None:
                import underlayer as U
                extra = U.parts(self, m, slab0)
            if L is not None:
                extra["isl_L"] = L
            if self._ff is not None:
                extra["film_fluct"] = self._ff
            if self._dom is not None:
                extra["domain"] = self._dom
            return mu, dict(sub=sub, isl=isl, fl=fl, G=ratio, T=m.T, D=m.D, t_b=getattr(m, "t_b", 0.0), **extra)
        return mu, sub, metal


class Model4g(Model4c):
    'Model4c whose metal scattering term is a q-dependent mix of two optical environments (phenomenological,'

    Lc = np.inf
    phiS = 0.0
    lc_n = 6

    def lc_weight(self):
        if not self.phiS > 0 or not np.isfinite(self.Lc):
            return None
        return self.phiS / (1.0 + (self.q * self.Lc / (2 * np.pi)) ** self.lc_n)

    def grains_sliced(self, m, slab0):
        out = super().grains_sliced(m, slab0)
        w = self.lc_weight()
        if w is None:
            return out
        L, F1 = self._islL, self._ff
        kept = {k: getattr(self, k, None) for k in ("_accL", "_void_part", "_bur_part", "_dom")}
        nu = self.nu_isl
        try:
            self.nu_isl = 0.0
            out0 = super().grains_sliced(m, slab0)
        finally:
            self.nu_isl = nu
        if L is not None:
            self._islL = (1 - w)[None, :] * L + w[None, :] * self._islL
        if F1 is not None and self._ff is not None:
            self._ff = (1 - w)[None, :] * F1 + w[None, :] * self._ff
        for k, a in kept.items():
            b = getattr(self, k, None)
            setattr(self, k, None if a is None or b is None else (1 - w)[None, :] * a + w[None, :] * b)
        return (1 - w)[None, :] * out + w[None, :] * out0


class GrainsC(GrainsB):
    'GrainsB whose base covers only the area fraction cb (the percolated, locally dense film; holes elsewhere).'

    def __init__(self, R, s, eta, fcap, omega, h_m, fb=0.0, rb=1.0, cb=1.0, **kw):
        self.cb = float(cb)
        super().__init__(R, s, eta, fcap, omega, h_m, fb=fb, rb=rb, **kw)
        self.t_b = self.fb * self.h_tot / (self.rb * self.cb)

    def slab_avg(self, metal, nslab=None, f1_au=M.AU_F1_12):
        'The laterally averaged stack: grain slices + base slices of density cb rb (same slicing as slab).'
        top, phi = Grains2.slab(self, metal, nslab, f1_au)
        if self.t_b <= 1e-6:
            return top, phi
        nb = M.SMOOTH_NB if M.SMOOTH else max(1, int(np.ceil(self.t_b / 2.0)))
        dm = dc(metal, f1_au=f1_au)
        r = self.cb * self.rb
        return top + [(r * dm, self.t_b / nb)] * nb, np.r_[phi, np.full(nb, r)]


class Model4d(Model4c):
    'Model4c with two optical environments (GrainsC): the grains and small islands (the grain-scale structure, q_y'

    def _cluster_term(self, m, slabA):
        nt = m.n_top
        E, z_up, wz = self._nodes(m, slabA)
        m.n_top = nt
        P = (E[:, 1:] * (E[:, 0] * wz)[:, None]).T
        acc = M.incoherent_sum(P, M.hemi_radius(z_up, m.L.Rj, m.L.Hj), self.q, m.nL * m.L.wj)
        return RE ** 2 * abs(rho_e(self.s.metal, self.f1)) ** 2 * acc

    def grains_sliced(self, m, slab0):
        if not isinstance(m, GrainsC) or m.fL <= 0:
            return super().grains_sliced(m, slab0)
        fL = m.fL
        m.fL = 0.0
        try:
            out = super().grains_sliced(m, slab0)
        finally:
            m.fL = fL
        slabA, _ = m.slab_avg(self.s.metal, f1_au=self.f1)
        return out + self._cluster_term(m, slabA)

    def substrate_ratio_avg(self, m, slab):
        nt = m.n_top if getattr(m, "n_top", None) is not None else len(slab)
        below = self._below()
        kzs, A, B, ts = stack(self.a_all, slab + below, self.d_sub)
        _, Ab, Bb, _ = stack(self.a_all, self._host(), self.d_sub)
        Eb, Ebare = A[len(slab) + 1] + B[len(slab) + 1], Ab[1] + Bb[1]
        P_bot, P_bare = Eb[0] * Eb[1:], Ebare[0] * Ebare[1:]
        rS = self._sub_rho(below)
        if len(slab) > nt:
            Et = A[nt + 1] + B[nt + 1]
            P_top, r_b = Et[0] * Et[1:], self.nu_isl * getattr(m, "cb", 1.0) * m.rb * rho_e(self.s.metal, self.f1)
        else:
            P_top, r_b = P_bot, 0.0
        return np.abs(rS * P_bot + r_b * (P_top - P_bot)) ** 2 / np.abs(rS * P_bare) ** 2

    def predict_grains(self, w, m, K, g=0.0, parts=False):
        if not isinstance(m, GrainsC):
            return super().predict_grains(w, m, K, g=g, parts=parts)
        if self.s.sub == "TiO2":
            raise NotImplementedError("base-layer model on TiO2")
        self.nu_sub = self.nu_isl
        slab0, _ = m.slab(self.s.metal, f1_au=self.f1)
        slabA0, phiA = m.slab_avg(self.s.metal, f1_au=self.f1)
        slabA = [(self.nu_isl * d, t) for d, t in slabA0]
        ratio = self.bg + (1 - self.bg) * self.substrate_ratio_avg(m, slabA)
        sub = w["cfac"] * ratio[None, :, None] * self.s.I_pre
        if g > 0:
            sub = sub * (1 + g * self.r_rep_grains(m))[None]
        isl = self.grains_sliced(m, slab0)
        Ise = M.layer_mid_int(self.a_all, slabA + self.sub_layers, self.d_sub)[: len(slabA)]
        fl = self.fluor(M.Morph(1.0, 1.0, 0.1, 0.3, m.h_tot), slabA, phiA, Ise) if m.h_tot > 0 else np.zeros(len(self.af))
        metal = self.smear(isl + fl[:, None])
        if self.Kaf is not None:
            sub = sub * (self.smear_ratio(ratio) / np.maximum(ratio, 1e-12))[None, :, None]
        mu = M.shadow_add(w, sub + K * metal[None], self)
        if parts:
            return mu, dict(sub=sub, isl=isl, fl=fl, G=ratio, T=m.T, D=m.D, t_b=m.t_b)
        return mu, sub, metal


class Model4e(Model4d):
    'Model4d with scale-dependent optics (hypothesis of 2026-11-02, after the cb test): the grains and small islands'

    nu_top = 1.0

    def _nodes(self, m, slab0):
        nt = m.n_top if getattr(m, "n_top", None) is not None else len(slab0)
        layers = [(self.nu_top * d, t) for d, t in slab0[:nt]] + [(self.nu_isl * d, t) for d, t in slab0[nt:]]
        a = np.r_[self.ai, np.maximum(self.af_c, 1e-4)]
        kzs, A, B, ts = stack(a, layers + self._below(), self.d_sub)
        E, zeta, wz = [], [], []
        top = 0.0
        for j in range(1, nt + 1):
            for u, w in zip(GL4_U, GL4_W):
                dz = u * ts[j]
                E.append(A[j] * np.exp(1j * kzs[j] * dz) + B[j] * np.exp(-1j * kzs[j] * dz))
                zeta.append(top + dz); wz.append(w * ts[j])
            top += ts[j]
        E = np.array(E); z_up = m.Hmax - np.array(zeta); wz = np.array(wz)
        return E, z_up, wz

    def _bare(self, m, slab):
        'The stack without the particle layer: its slices become vacuum (same node positions).'
        nt = m.n_top
        return [(0j, t) for _, t in slab[:nt]] + list(slab[nt:])

    def _cluster_term(self, m, slabA):
        return super()._cluster_term(m, self._bare(m, slabA))

    def predict_grains(self, w, m, K, g=0.0, parts=False):
        if not isinstance(m, GrainsC):
            raise TypeError("Model4e needs GrainsC")
        if self.s.sub == "TiO2":
            raise NotImplementedError("base-layer model on TiO2")
        self.nu_sub = self.nu_isl
        slab0, _ = m.slab(self.s.metal, f1_au=self.f1)
        slabA0, phiA = m.slab_avg(self.s.metal, f1_au=self.f1)
        nt = m.n_top
        slabA = [(self.nu_top * d, t) for d, t in slabA0[:nt]] + [(self.nu_isl * d, t) for d, t in slabA0[nt:]]
        ratio = self.bg + (1 - self.bg) * self.substrate_ratio_avg(m, self._bare(m, slabA))
        sub = w["cfac"] * ratio[None, :, None] * self.s.I_pre
        if g > 0:
            sub = sub * (1 + g * self.r_rep_grains(m))[None]
        isl = self.grains_sliced(m, slab0)
        Ise = M.layer_mid_int(self.a_all, slabA + self.sub_layers, self.d_sub)[: len(slabA)]
        fl = self.fluor(M.Morph(1.0, 1.0, 0.1, 0.3, m.h_tot), slabA, phiA, Ise) if m.h_tot > 0 else np.zeros(len(self.af))
        metal = self.smear(isl + fl[:, None])
        if self.Kaf is not None:
            sub = sub * (self.smear_ratio(ratio) / np.maximum(ratio, 1e-12))[None, :, None]
        mu = M.shadow_add(w, sub + K * metal[None], self)
        if parts:
            return mu, dict(sub=sub, isl=isl, fl=fl, G=ratio, T=m.T, D=m.D, t_b=m.t_b)
        return mu, sub, metal

class Model4f(Model4e):
    'Model4e with a free weight nu_L of the particle-layer medium seen by the large flat clusters and the substrate'

    nu_L = 0.0

    def _bare(self, m, slab):
        nt = m.n_top
        return [(self.nu_L * d, t) for d, t in slab[:nt]] + list(slab[nt:])
