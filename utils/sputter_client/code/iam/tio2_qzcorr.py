'IAM_TIO2_QZCORR=1 (2026-11-18): local-field correction of the buried TiO2 island term.'
from __future__ import annotations

import os

import numpy as np

import iam as M

ON = os.environ.get("IAM_TIO2_QZCORR", "0") == "1"


def sphere_amp(R, qpar, Qz):
    'Sphere amplitude about its centre; R (nR,), qpar (nq,), Qz (..., na) complex -> (..., nR, na, nq).'
    K2 = np.asarray(Qz, complex)[..., None, :, None] ** 2 + np.asarray(qpar, float)[None, None, :] ** 2
    x = np.asarray(R, float)[:, None, None] * np.sqrt(K2)
    small = np.abs(x) < 1e-3
    xs = np.where(small, 1.0, x)
    f = np.where(small, 1 - x * x / 10, 3 * (np.sin(xs) - xs * np.cos(xs)) / xs ** 3)
    return (4 / 3 * np.pi * np.asarray(R, float) ** 3)[:, None, None] * f


def _slices(mdl, phm, with_cap):
    dm = M.dc(mdl.s.metal, f1_au=mdl.f1)
    sl = [(d + mdl.nu_sub * p * dm, t) for (d, t), p in zip(mdl.sub_layers, phm)]
    if with_cap:
        return [(mdl.nu_sub * mdl.cap[0] * dm, mdl.cap[1])] + sl, 1, dm
    return sl, 0, dm


def factor(mdl, phm, wt, regions_area, pop, rows, cols):
    'C on the coarse grid (alpha_f rows `rows` of mdl.af_c, q columns `cols`), shape (len(rows), len(cols)).'
    af = mdl.af_c[rows]
    q = mdl.q[cols]
    a_all = np.r_[mdl.ai, np.maximum(af, 1e-4)]
    if M.QUAD == "gh":
        Rj, Hj, wj = pop.Rj, pop.Hj, pop.wj
    else:
        Rj, wj = M.ln_nodes(pop.R, pop.s, None, None, dr=M.QUAD_DR_ORIENT, rel=M.QUAD_REL_ORIENT)
        Hj = pop.a * Rj
    Req = np.cbrt(np.asarray(Rj) ** 2 * np.asarray(Hj) / 2.0)
    wj = np.asarray(wj, float)
    qzv = M.K0 * (np.sin(np.radians(mdl.ai)) + np.sin(np.radians(np.maximum(af, 0.0))))
    Fv2 = np.abs(sphere_amp(Req, q, qzv[None, :])[0]) ** 2
    num = np.zeros((len(af), len(q)))
    den = np.zeros((len(af), len(q)))
    n = len(phm)
    keep = np.asarray(wt, float) > 1e-12 * max(float(np.max(wt)), 1e-300)
    for area, with_cap in regions_area:
        layers, off, dm = _slices(mdl, phm, with_cap)
        kzs, A, B, ts = M.stack(a_all, layers, mdl.d_sub)
        idx = [off + 1 + s for s in range(n) if keep[s]]
        k = np.array([kzs[i] for i in idx])
        t = np.array([ts[i] for i in idx])[:, None]
        Cd = np.array([A[i] for i in idx]) * np.exp(1j * k * 0.5 * t)
        Cu = np.array([B[i] for i in idx]) * np.exp(-1j * k * 0.5 * t)
        d_s = np.array([layers[i - 1][0] for i in idx])
        v = np.asarray(wt, float)[keep]
        c = np.abs(dm - d_s) ** 2 / abs(dm) ** 2
        Di, Ui, ki = Cd[:, :1], Cu[:, :1], k[:, :1]
        Df, Uf, kf = Cd[:, 1:], Cu[:, 1:], k[:, 1:]
        EE = np.abs((Di + Ui) * (Df + Uf)) ** 2
        den += area * np.einsum("s,sa,j,jaq->aq", v, EE, wj, Fv2)
        for s in range(len(v)):
            Amp = (Di[s] * Df[s])[None, :, None] * sphere_amp(Req, q, (ki[s] + kf[s])[None, :])[0] \
                + (Ui[s] * Df[s])[None, :, None] * sphere_amp(Req, q, (-ki[s] + kf[s])[None, :])[0] \
                + (Di[s] * Uf[s])[None, :, None] * sphere_amp(Req, q, (ki[s] - kf[s])[None, :])[0] \
                + (Ui[s] * Uf[s])[None, :, None] * sphere_amp(Req, q, (-ki[s] - kf[s])[None, :])[0]
            num += area * v[s] * c[s] * np.einsum("j,jaq->aq", wj, np.abs(Amp) ** 2)
    return num / np.maximum(den, 1e-300)


def factor_top(mdl, phm, pop, rows, cols):
    'C_top for the islands on the cap (IAM_CAP_PLACE=on: production |E_i(0)|^2 |E_f(0)|^2 |F(q_vac)|^2): the four'
    af = mdl.af_c[rows]
    q = mdl.q[cols]
    a_all = np.r_[mdl.ai, np.maximum(af, 1e-4)]
    if M.QUAD == "gh":
        Rj, Hj, wj = pop.Rj, pop.Hj, pop.wj
    else:
        Rj, wj = M.ln_nodes(pop.R, pop.s, None, None, dr=M.QUAD_DR_ORIENT, rel=M.QUAD_REL_ORIENT)
        Hj = pop.a * Rj
    Req = np.cbrt(np.asarray(Rj) ** 2 * np.asarray(Hj) / 2.0)
    wj = np.asarray(wj, float)
    layers, _, _ = _slices(mdl, phm, True)
    kzs, A, B, ts = M.stack(a_all, layers, mdl.d_sub)
    r = np.asarray(B[0]) / np.asarray(A[0])
    ri, rf = r[0], r[1:]
    ki = M.K0 * np.sin(np.radians(mdl.ai))
    kf = M.K0 * np.sin(np.radians(np.maximum(af, 0.0)))
    terms = [(1.0, ki + kf), (ri, -ki + kf), (rf, ki - kf), (ri * rf, -ki - kf)]
    num = 0
    for c, Q in terms:
        ph = np.exp(-1j * np.asarray(Q)[None, :] * Req[:, None])
        cc = np.asarray(c)[None, :, None] if np.ndim(c) else c
        num = num + cc * ph[:, :, None] * sphere_amp(Req, q, np.asarray(Q)[None, :])[0]
    num = np.einsum("j,jaq->aq", wj, np.abs(num) ** 2)
    den = (np.abs(1 + ri) ** 2 * np.abs(1 + rf) ** 2)[:, None] \
        * np.einsum("j,jaq->aq", wj, np.abs(sphere_amp(Req, q, (ki + kf)[None, :])[0]) ** 2)
    return num / np.maximum(den, 1e-300)


def _to_full(mdl, C):
    lo = np.log(np.maximum(C, 1e-300))
    af, q = mdl.af_c[mdl.ia], mdl.q[mdl.iq]
    A1 = np.array([np.interp(mdl.af_c, af, lo[:, j]) for j in range(lo.shape[1])]).T
    A2 = np.array([np.interp(np.log(mdl.q), np.log(q), A1[i]) for i in range(A1.shape[0])])
    return np.exp(A2)


def factor_top_full(mdl, phm, pop):
    return _to_full(mdl, factor_top(mdl, phm, pop, mdl.ia, mdl.iq))


def factor_full(mdl, phm, wt, regions_area, pop):
    'C on the full (alpha_f, q) grid: coarse rows / columns of the production Born, log-interpolated like it.'
    C = factor(mdl, phm, wt, regions_area, pop, mdl.ia, mdl.iq)
    lo = np.log(np.maximum(C, 1e-300))
    af, q = mdl.af_c[mdl.ia], mdl.q[mdl.iq]
    A1 = np.array([np.interp(mdl.af_c, af, lo[:, j]) for j in range(lo.shape[1])]).T
    A2 = np.array([np.interp(np.log(mdl.q), np.log(q), A1[i]) for i in range(A1.shape[0])])
    return np.exp(A2)
