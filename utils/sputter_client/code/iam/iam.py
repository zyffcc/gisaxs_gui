'IAM v0: in-situ anchored forward model for sputter GISAXS (protocol _iam/IAM_v0_protocol.json).'

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import iam_paths as IAMP

import numpy as np
from scipy.special import j1, xlogy

sys.path.insert(0, str(IAMP.GIMAP))
from utils.sputter_forward.data import Result
from utils.sputter_forward.grid import NativeGrid, native_grid

HERE = Path(__file__).resolve().parent
FV2 = HERE.parent
RAW = IAMP.RAW


LAM_12 = 0.103320165
LAM = float(os.environ.get("IAM_LAM_NM", repr(LAM_12)))
K0 = 2 * np.pi / LAM
RE = 2.8179403262e-6
NA21 = 602.214076


FL_KEV = float(os.environ.get("IAM_FL_KEV", "9.70"))
LAM_FL = 1.239841984 / FL_KEV
K_FL = 2 * np.pi / LAM_FL


def pref(lam):
    return RE * lam ** 2 / (2 * np.pi)


N_AU, N_AG = 19.32 / 196.967 * NA21, 10.49 / 107.868 * NA21
N_TIO2, N_SI = 3.9 / 79.866 * NA21, 2.329 / 28.086 * NA21
OPT = {
    "Ag": ((pref(LAM) * N_AG * 47.0, pref(LAM) * N_AG * 2.15), (pref(LAM_FL) * N_AG * 47.0, pref(LAM_FL) * N_AG * 3.5)),
    "TiO2": ((pref(LAM) * N_TIO2 * 38.2, 1.34e-7), (pref(LAM_FL) * N_TIO2 * 38.2, 2.96e-7)),
    "Si": ((pref(LAM) * N_SI * 14.05, 4.16e-8), (pref(LAM_FL) * N_SI * 14.05, 9.2e-8)),
}


POLY = {"PS": (1.05, 56, 104.15, 1.2e-9), "PET": (1.38, 100, 192.17, 2.0e-9),
        "PSBMA": (1.30, 150, 279.35, 6.0e-9), "PNO": (1.20, 94, 173.21, 2.5e-9),
        "PEO": (1.13, 24, 44.05, 2.0e-9)}
for _n, (_rho, _z, _mw, _b) in POLY.items():
    _ne = _rho / _mw * _z * NA21
    OPT[_n] = ((pref(LAM) * _ne, _b), (pref(LAM_FL) * _ne, 1.9 * _b))
AU_F1_12, AU_F2_12, AU_F1_FL, AU_F2_FL = 64.0, 9.7, 70.0, 5.0
MU_AU_12 = 4 * np.pi * pref(LAM) * N_AU * AU_F2_12 / LAM
P_L3, OMEGA_L3, EFF_RATIO = 0.58, 0.32, 1.1


N_CU = 8.96 / 63.546 * NA21
LAM_CU, FL_KEV_CU = 0.105, 8.048
CU_LAM_TOL = 0.05

CU_F1, CU_F2, CU_F1_FL, CU_F2_FL = 28.2, 2.45, 27.04, 0.589
OPT["Cu"] = ((pref(LAM) * N_CU * CU_F1, pref(LAM) * N_CU * CU_F2), (pref(LAM_FL) * N_CU * CU_F1_FL, pref(LAM_FL) * N_CU * CU_F2_FL))
MU_CU = 4 * np.pi * pref(LAM) * N_CU * CU_F2 / LAM
P_K_CU, OMEGA_K_CU, EFF_RATIO_CU = 0.873, 0.445, 1.26


OPTICS = os.environ.get("IAM_OPTICS", "const")
if OPTICS not in ("const", "table", "anchored"):
    raise SystemExit(f"IAM_OPTICS={OPTICS}: const, table or anchored")
ANCHOR_NM = float(os.environ.get("IAM_OPTICS_ANCHOR_NM", repr(LAM_12)))
E_EV, E_FL_EV = 1239.841984 / LAM, 1000.0 * FL_KEV
if OPTICS == "table":
    import xray_optics as XO
    for _name in list(OPT):
        _d, _f = XO.decrement(_name, E_EV), XO.decrement(_name, E_FL_EV)
        OPT[_name] = ((_d.real, -_d.imag), (_f.real, -_f.imag))
    AU_F1_12, AU_F2_12 = XO.f12("Au", E_EV)
    AU_F1_FL, AU_F2_FL = XO.f12("Au", E_FL_EV)
    MU_AU_12 = 4 * np.pi * pref(LAM) * N_AU * AU_F2_12 / LAM
    CU_F1, CU_F2 = XO.f12("Cu", E_EV)
    CU_F1_FL, CU_F2_FL = XO.f12("Cu", E_FL_EV)
    OPT["Cu"] = ((pref(LAM) * N_CU * CU_F1, pref(LAM) * N_CU * CU_F2),
                 (pref(LAM_FL) * N_CU * CU_F1_FL, pref(LAM_FL) * N_CU * CU_F2_FL))
    MU_CU = 4 * np.pi * pref(LAM) * N_CU * CU_F2 / LAM
    if not XO.fluoresces("Au", E_EV):
        P_L3 = 0.0
    if not XO.fluoresces("Cu", E_EV):
        P_K_CU = 0.0
if OPTICS == "anchored":
    import xray_optics as XO

    def _hand_beam(lam):
        '(delta, beta) of the beam channel at wavelength lam, built as the hand dictionary above builds them.'
        out = {"Ag": (pref(lam) * N_AG * 47.0, pref(lam) * N_AG * 2.15), "TiO2": (pref(lam) * N_TIO2 * 38.2, 1.34e-7),
               "Si": (pref(lam) * N_SI * 14.05, 4.16e-8)}
        for _n, (_rho, _z, _mw, _b) in POLY.items():
            out[_n] = (pref(lam) * (_rho / _mw * _z * NA21), _b)
        return out
    E_ANCHOR = 1239.841984 / ANCHOR_NM
    for _name, (_d0, _b0) in _hand_beam(ANCHOR_NM).items():
        _ta, _te = XO.decrement(_name, E_ANCHOR), XO.decrement(_name, E_EV)
        OPT[_name] = ((_d0 * (_te.real / _ta.real), _b0 * (_te.imag / _ta.imag)), OPT[_name][1])
    _fa, _fe = XO.f12("Au", 1239.841984 / LAM_12), XO.f12("Au", E_EV)
    AU_F1_12, AU_F2_12 = 64.0 * (_fe[0] / _fa[0]), 9.7 * (_fe[1] / _fa[1])
    MU_AU_12 = 4 * np.pi * pref(LAM) * N_AU * AU_F2_12 / LAM
    _fa, _fe = XO.f12("Cu", 1239.841984 / LAM_CU), XO.f12("Cu", E_EV)
    CU_F1, CU_F2 = 28.2 * (_fe[0] / _fa[0]), 2.45 * (_fe[1] / _fa[1])
    OPT["Cu"] = ((pref(LAM) * N_CU * CU_F1, pref(LAM) * N_CU * CU_F2), OPT["Cu"][1])
    MU_CU = 4 * np.pi * pref(LAM) * N_CU * CU_F2 / LAM
    if not XO.fluoresces("Au", E_EV):
        P_L3 = 0.0
    if not XO.fluoresces("Cu", E_EV):
        P_K_CU = 0.0


AI_OVERRIDE = os.environ.get("IAM_AI")
LAM_REMAP = os.environ.get("IAM_LAM_REMAP", "0") == "1"


def af_remap(a, af, lam_from, lam_to=None):
    'An array (..., n_af, n_q) on the alpha_f centres af (deg) measured at lam_from, moved to lam_to (default LAM)'
    lam_to = LAM if lam_to is None else lam_to
    a = np.asarray(a, float)
    if abs(lam_from - lam_to) < 1e-12:
        return a.copy()
    af = np.asarray(af, float)
    src = np.degrees(np.arcsin(np.clip(lam_from / lam_to * np.sin(np.radians(af)), -1.0, 1.0)))
    flat = np.moveaxis(a, -2, -1).reshape(-1, len(af))
    out = np.array([np.interp(src, af, row, left=0.0, right=0.0) for row in flat])
    return np.moveaxis(out.reshape(np.moveaxis(a, -2, -1).shape), -1, -2)


def fluorescence_on(metal):
    "The metal's fluorescence line is excited at the beam energy (always with the hand constants: 12 keV Au, Cu"
    if OPTICS != "table":
        return True
    return XO.fluoresces(metal, E_EV)


def dc(name, fl=False, f1_au=AU_F1_12):
    'Complex optical decrement d = delta - i beta (n = 1 - d).'
    if name == "Au":
        if fl:
            return pref(LAM_FL) * N_AU * (AU_F1_FL - 1j * AU_F2_FL)
        return pref(LAM) * N_AU * (f1_au - 1j * AU_F2_12)
    d, b = OPT[name][1 if fl else 0]
    return d - 1j * b


def rho_e(name, f1_au=AU_F1_12):
    return dc(name, f1_au=f1_au) / pref(LAM)


def kz(alpha_deg, d, k):
    s = np.sin(np.radians(alpha_deg))
    v = np.sqrt(s * s - 2 * d + 0j)
    return k * np.where(v.imag < 0, -v, v)


def stack(alpha_deg, layers, d_sub, k=K0):
    'layers [(d, t)] from the top. Returns kzs, A (down), B (up) at the top of each layer (index 0 vacuum at z=0,'
    a = np.atleast_1d(np.asarray(alpha_deg, float))
    ds = [0j] + [d for d, _ in layers] + [d_sub]
    ts = [0.0] + [float(t) for _, t in layers] + [0.0]
    kzs = [kz(a, d, k) for d in ds]
    n = len(ds)
    X = [None] * n
    X[-1] = np.zeros_like(kzs[-1])
    for j in range(n - 2, -1, -1):
        r = (kzs[j] - kzs[j + 1]) / (kzs[j] + kzs[j + 1])
        xb = (r + X[j + 1]) / (1 + r * X[j + 1])
        X[j] = xb * np.exp(2j * kzs[j] * ts[j])
    A = [None] * n; B = [None] * n
    A[0] = np.ones_like(kzs[0]); B[0] = X[0]
    for j in range(0, n - 1):
        ph = np.exp(1j * kzs[j] * ts[j])
        xbot = X[j] * np.exp(-2j * kzs[j] * ts[j])
        A[j + 1] = A[j] * ph * (1 + xbot) / (1 + X[j + 1])
        B[j + 1] = A[j + 1] * X[j + 1]
    return kzs, A, B, ts


def layer_mid_int(alpha_deg, layers, d_sub, k=K0):
    '|E|^2 at the middle of every layer 1..N and at the substrate top: array (N+1, n_alpha).'
    kzs, A, B, ts = stack(alpha_deg, layers, d_sub, k)
    out = []
    for j in range(1, len(layers) + 1):
        z = 0.5 * ts[j]
        out.append(np.abs(A[j] * np.exp(1j * kzs[j] * z) + B[j] * np.exp(-1j * kzs[j] * z)) ** 2)
    out.append(np.abs(A[-1]) ** 2)
    return np.array(out)


def base_amps(alpha_deg, layers, d_sub, j, k=K0):
    kzs, A, B, ts = stack(alpha_deg, layers, d_sub, k)
    t = ts[j]
    return A[j] * np.exp(1j * kzs[j] * t), B[j] * np.exp(-1j * kzs[j] * t), kzs[j]


GL_S, GL_W = np.polynomial.legendre.leggauss(20)
GL_S, GL_W = 0.5 * (GL_S + 1), 0.5 * GL_W
GL_R = np.sqrt(1 - GL_S ** 2)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(7)
GH_W = GH_W / GH_W.sum()


QUAD = os.environ.get("IAM_QUAD", "dense")
QUAD_DR = float(os.environ.get("IAM_QUAD_DR", "0.6"))
QUAD_REL = float(os.environ.get("IAM_QUAD_REL", "0.12"))
QUAD_XMAX = 3.5


SMOOTH = os.environ.get("IAM_SMOOTH", "0") == "1"

NFEV_MULT = float(os.environ.get("IAM_NFEV_MULT", "1"))
SMOOTH_NSLAB = int(os.environ.get("IAM_NSLAB", "96"))
SMOOTH_NZ = int(os.environ.get("IAM_NZ", "384"))
SMOOTH_NB = int(os.environ.get("IAM_NB", "16"))


QUAD_DR_ORIENT = float(os.environ.get("IAM_QUAD_DR_ORIENT", "2.0"))
QUAD_REL_ORIENT = float(os.environ.get("IAM_QUAD_REL_ORIENT", "0.3"))


def ln_nodes(R, s, X, W, dr=None, rel=None):
    'Radii R_j and number weights w_j (sum 1) of R_j = R exp(s x - s^2 / 2), x ~ N(0, 1). X, W: the legacy'
    R, s = float(R), float(s)
    if QUAD == "gh" or s < 1e-4:
        return R * np.exp(s * np.asarray(X) - 0.5 * s * s), np.asarray(W)
    dR = min(QUAD_DR if dr is None else dr, (QUAD_REL if rel is None else rel) * R)
    lo, hi = R * np.exp(-s * QUAD_XMAX - 0.5 * s * s), R * np.exp(s * QUAD_XMAX - 0.5 * s * s)

    def grid(n):
        Rj = np.linspace(lo, hi, n)
        x = (np.log(Rj / R) + 0.5 * s * s) / s
        w = np.exp(-0.5 * x * x) / Rj
        w[[0, -1]] *= 0.5
        return Rj, w / w.sum()
    u = (hi - lo) / dR
    if not SMOOTH:
        return grid(int(min(max(np.ceil(u) + 1, 9), 4000)))


    if u + 1 <= 9:
        return grid(9)
    k = min(int(np.floor(u)), 3998)
    f = float(u - np.floor(u)) if k < 3998 else 0.0
    Ra, wa = grid(k + 1)
    if f < 1e-12:
        return Ra, wa
    Rb, wb = grid(k + 2)
    return np.r_[Ra, Rb], np.r_[(1.0 - f) * wa, f * wb]


def jinc(x):
    x = np.asarray(x, float)
    out = np.full(x.shape, 0.5)
    m = np.abs(x) > 1e-6
    out[m] = j1(x[m]) / x[m]
    return out


def ff_sep(qpar, Qz, R, H):
    'Hemispheroid F(qpar_j, Qz_i) for vectors qpar (nq) and Qz (na, complex): (na, nq).'
    g = (1 - GL_S ** 2) * jinc(np.asarray(qpar)[:, None] * R * GL_R) * GL_W
    e = np.exp(1j * np.asarray(Qz)[:, None] * H * GL_S)
    return 2 * np.pi * R * R * H * (e @ g.T)


def ff_sep_sum(qpar, Qz, R, H, c):
    'sum_j c_j ff_sep(qpar, Qz, R_j, H_j) for arrays R, H, c (vectorised over the size nodes).'
    R, H, c = (np.asarray(v, float) for v in (R, H, c))
    g = (1 - GL_S ** 2) * jinc(np.asarray(qpar)[None, :, None] * (R[:, None] * GL_R[None, :])[:, None, :]) * GL_W
    e = np.exp(1j * np.asarray(Qz)[None, :, None] * (H[:, None] * GL_S[None, :])[:, None, :])
    return np.einsum("j,jas,jqs->aq", 2 * np.pi * c * R * R * H, e, g)


def ff_arr_nodes(qpar, Qz, R, H):
    'ff_arr for size-node vectors R, H (n,) on equal-shape arrays qpar, Qz: (n, *qpar.shape).'
    R = np.asarray(R, float).reshape((-1,) + (1,) * (np.ndim(qpar) + 1))
    H = np.asarray(H, float).reshape(R.shape)
    g = (1 - GL_S ** 2) * jinc(qpar[None, ..., None] * R * GL_R) * GL_W
    e = np.exp(1j * Qz[None, ..., None] * H * GL_S)
    return 2 * np.pi * (R * R * H)[..., 0] * np.sum(g * e, axis=-1)


def sig_disk(r, q):
    '2 pi r^2 jinc(q r) for radii r (any shape): (..., nq).'
    r = np.asarray(r, float)
    return 2 * np.pi * r[..., None] ** 2 * jinc(np.asarray(q) * r[..., None])


def incoherent_sum(P, r, q, c, S=None, chunk=48):
    'sum_j c_j |sum_n P[a, n] sig_disk(r[j, n], q)|^2 S[j, q] -> (n_af, nq). P (n_af, nodes) complex field-product'
    q = np.asarray(q)
    na, nq = P.shape[0], len(q)
    out = np.zeros((na, nq))
    c = np.asarray(c, float)
    for j0 in range(0, len(c), chunk):
        rr = r[j0:j0 + chunk]
        act = np.any(rr > 0, axis=0)
        if not act.any():
            continue
        Sig = sig_disk(rr[:, act], q)
        nc, nact = Sig.shape[0], Sig.shape[1]
        A = (P[:, act] @ Sig.transpose(1, 0, 2).reshape(nact, nc * nq)).reshape(na, nc, nq)
        I = A.real ** 2 + A.imag ** 2
        if S is not None:
            I = I * S[None, j0:j0 + chunk]
        out += np.einsum("ajq,j->aq", I, c[j0:j0 + chunk])
    return out


def coherent_sig(r, q, c):
    'sum_j c_j sig_disk(r[j, z], q) -> (nz, nq) (linear in the amplitude: sizes summed before the depth integral).'
    q = np.asarray(q)
    out = np.zeros((r.shape[1], len(q)))
    act = np.any(r > 0, axis=0)
    if act.any():
        out[act] = np.einsum("j,jzq->zq", np.asarray(c, float), sig_disk(r[:, act], q))
    return out


def hemi_radius(z, Rj, Hj):
    'Radius of hemispheroids (R_j, H_j) at heights z: (n_size, nz).'
    z = np.asarray(z, float)[None, :]
    Rj = np.asarray(Rj, float)[:, None]
    Hj = np.asarray(Hj, float)[:, None]
    return np.where((z < Hj) & (z >= 0), Rj * np.sqrt(np.clip(1 - (z / Hj) ** 2, 0, None)), 0.0)


def ff_arr(qpar, Qz, R, H):
    'Hemispheroid F for arrays of equal shape (real Qz).'
    g = (1 - GL_S ** 2) * jinc(qpar[..., None] * R * GL_R) * GL_W
    e = np.exp(1j * Qz[..., None] * H * GL_S)
    return 2 * np.pi * R * R * H * np.sum(g * e, axis=-1)


def s_para(q, D, omega):
    F = np.exp(-0.5 * (q * omega * D) ** 2)
    return (1 - F * F) / np.maximum(1 - 2 * F * np.cos(q * D) + F * F, 1e-12)


SF = os.environ.get("IAM_SF", "para")
if SF not in ("para", "hd"):
    raise SystemExit(f"IAM_SF={SF!r}: 'para' or 'hd'")
HD_ETA0, HD_ETA_LIM = 0.67, (0.05, 0.62)
_HD_P = 7.0 / 3.0 - 4.0 * np.sqrt(3.0) / np.pi
_HD_X, _HD_W = np.polynomial.legendre.leggauss(64)
_HD_X, _HD_W = 0.5 * (_HD_X + 1.0), 0.5 * _HD_W


def s_hard_disc(y, eta):
    'Structure factor of a hard-disc fluid at y = q sigma (sigma: core diameter) for the packing fraction eta:'
    from scipy.special import j0
    y = np.asarray(y, float)
    num = 1.0 + eta + 3.0 * _HD_P * eta ** 2 - _HD_P * eta ** 3
    c0 = -num / (1.0 - eta) ** 3
    s2 = 3.0 * eta ** 2 / 8.0 * (8.0 * (1.0 - 2.0 * _HD_P) + (25.0 - 9.0 * _HD_P) * _HD_P * eta
                                 - (7.0 - 3.0 * _HD_P) * _HD_P * eta ** 2) / num
    h = _HD_X / 2.0
    w2 = 2.0 * (np.arccos(h) - h * np.sqrt(1.0 - h * h)) / np.pi
    cx = c0 * (1.0 + 4.0 * eta * (w2 - 1.0) + s2 * _HD_X)
    cq = 2.0 * np.pi * np.tensordot(j0(y[..., None] * _HD_X), _HD_W * _HD_X * cx, axes=([-1], [0]))
    return 1.0 / (1.0 - 4.0 * eta / np.pi * cq)


def hd_eta(omega):
    'Packing fraction of the hard cores for the disorder parameter omega (IAM_SF=hd).'
    return float(np.clip(HD_ETA0 - float(omega), HD_ETA_LIM[0], HD_ETA_LIM[1]))


def s_lat(q, D, omega):
    'Interference function of grains at the interference distance D: s_para (default) or, with IAM_SF=hd, the'
    if SF == "hd":
        eta = hd_eta(omega)
        return s_hard_disc(np.asarray(q, float) * np.asarray(D, float) * np.sqrt(2.0 * np.sqrt(3.0) * eta / np.pi), eta)
    return s_para(q, D, omega)


def lma_D(D, Rj, Rm):
    'Spacing seen by size class j in the local monodisperse sum: D R_j / R_m (default, spacing ~ size) or, with'
    if os.environ.get("IAM_SQ", "lma") in ("common", "free"):
        return np.full_like(np.asarray(Rj, dtype=float), D, dtype=float)
    return D * Rj / Rm


class Morph:
    'R (nm), a = H / R, s (lognormal sigma of R), omega; mass thickness h_m (nm).'

    def __init__(self, R, a, s, omega, h_m, kD=1.07):
        self.R, self.a, self.s, self.omega, self.h_m = float(R), float(a), float(s), float(omega), float(h_m)
        self.Rj, self.wj = ln_nodes(self.R, self.s, GH_X, GH_W)
        self.Hj = self.a * self.Rj
        self.V = (2 * np.pi / 3) * self.Rj ** 2 * self.Hj
        self.n = self.h_m / float(np.sum(self.wj * self.V))
        self.D = kD / np.sqrt(self.n)
        self.coverage = float(self.n * np.sum(self.wj * np.pi * self.Rj ** 2))
        self.Hmax = float(np.max(self.Hj))

    def phi_profile(self, z_up):
        z = np.asarray(z_up, float)[..., None]
        r2 = np.where(z < self.Hj, self.Rj ** 2 * (1 - (z / self.Hj) ** 2), 0.0)
        return self.n * np.sum(self.wj * np.pi * r2, axis=-1)

    def slab(self, metal, nslab=6, f1_au=AU_F1_12, fl=False):
        'Effective-medium slices of the island layer from the top: [(d, t)], metal fraction per slice.'
        H = self.Hmax
        edges = np.linspace(H, 0, nslab + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        phi = np.clip(self.phi_profile(mids), 0, 1)
        dm = dc(metal, fl=fl, f1_au=f1_au)
        return [(p * dm, H / nslab) for p in phi], phi


def frame_clock(rows):
    return np.array([datetime.fromisoformat(r["timestamp"]).timestamp() for r in rows])


def scan_grid():
    "The toolkit's native grid; on a machine without the toolkit (Maxwell bundle, maxwell_bundle.py) the edges cached"
    try:
        return native_grid()
    except Exception:
        p = IAMP.DATA / "native_grid.npz"
        if not p.exists():
            raise
        with np.load(p) as z:
            return NativeGrid(z["native_edges"], z["q_edges"])


def dat_motion(growth_scan):
    f = RAW / "online" / f"{growth_scan}.dat"
    c = IAMP.DATA / "motion" / f"{growth_scan}.npz"
    if not f.exists() and c.exists():
        with np.load(c) as z:
            return z["T"], z["X"]
    lines = f.read_text(errors="replace").splitlines()
    T, X = [], []
    for ln in lines:
        p = [v.strip() for v in ln.split(",")]
        if len(p) == 4:
            try:
                T.append(float(p[1])); X.append(float(p[2]))
            except ValueError:
                pass
    return np.array(T), np.array(X)


def pixmask_of(label):
    'sha256 of the detector pixel mask a cache was built with (DATA/<label>/pixmask.json, written by remask_caches.py),'
    def rd(p):
        return json.loads(p.read_text(encoding="utf-8"))["sha256"] if p.exists() else None
    root, lab = rd(IAMP.DATA / "pixmask.json"), rd(IAMP.DATA / label / "pixmask.json")
    if root != lab:
        raise ValueError(f"{label}: cache mask {lab} differs from the DATA root mask {root}")
    want = os.environ.get("IAM_PIXMASK")
    if want and not (lab or "none").startswith(want):
        raise ValueError(f"{label}: cache mask {lab}, IAM_PIXMASK requires {want}")
    return lab


def warm_starts(f, label, t, names):
    'Start vectors from stored window solutions WORK/<branch>/<label>_<t>.json of the branches in IAM_WARM (comma'
    out = []
    for br in filter(None, os.environ.get("IAM_WARM", "").split(",")):
        wp = IAMP.WORK / br / f"{label}_{t:g}.json"
        if not wp.exists() and os.environ.get("IAM_WARM_REQUIRE") == "1":
            raise FileNotFoundError(f"warm start {wp}")
        if wp.exists():
            prm = json.loads(wp.read_text(encoding="utf-8"))["params"]
            out.append((br, f.x0(**{k: v for k, v in prm.items() if k in names})))
    return out


def yscan_profile(label):
    d = IAMP.DATA / f"{label}_yb"
    if not (d / "intake" / "run.json").exists():
        return None
    pixmask_of(f"{label}_yb")
    run = json.loads((d / "intake" / "run.json").read_text(encoding="utf-8"))
    Y, E = [], []
    for c in run["chunks"]:
        z = np.load(d / "intake" / c["file"])
        meta = json.loads((d / "intake" / Path(c["file"]).with_suffix(".json")).read_text(encoding="utf-8"))
        ex = np.array([m["exposure_s"] for m in meta["rows"]])
        Y.append(z["Y"].astype(float).sum((1, 2, 3))); E.append(z["P"].astype(float).sum((1, 2, 3)) * ex)
    rate = np.concatenate(Y) / np.concatenate(E)
    pos = json.loads((d / "positions.json").read_text(encoding="utf-8"))
    rows = np.asarray(pos["rows"]); cols = pos["columns"]
    y = rows[:, cols.index("POS")][: len(rate)]; mon = rows[:, cols.index("QBPM_EH1")][: len(rate)]
    r = rate / (mon / np.median(mon))
    return y, r / np.median(r)


SCANS = {}


SHADOW_BG = os.environ.get("IAM_SHADOW_BG", "0")
SHADOW_AB = float(os.environ.get("IAM_SHADOW_AB", "1"))
if SHADOW_BG not in ("0", "flat", "pre0"):
    raise ValueError(f"IAM_SHADOW_BG={SHADOW_BG}: use flat or pre0 (unset / 0: off)")
_SHADOW_WARNED = []


def shadow_term(w, mdl=None):
    'The additive measured background of IAM_SHADOW_BG in the window w (rate, broadcastable to mu), None without it.'
    if "shadow_add" in w:
        return w["shadow_add"]
    if "shadow_pre0" in w:
        if mdl is None or not hasattr(mdl, "bg"):
            raise NotImplementedError("IAM_SHADOW_BG=pre0 is defined for the iam4.Model4c family only (bg I_pre)")
        return w["shadow_pre0"][:, None, :] * mdl.bg * mdl.s.I_pre
    return None


def shadow_add(w, mu, mdl=None):
    'mu + the IAM_SHADOW_BG term; mu itself (same object) without it. A non-positive result (dS < 0 at low mu) is'
    a = shadow_term(w, mdl)
    if a is None:
        return mu
    out = mu + a
    if not _SHADOW_WARNED and np.any(out <= 0):
        _SHADOW_WARNED.append(1)
        import warnings
        warnings.warn(f"IAM_SHADOW_BG={SHADOW_BG}: mu <= 0 in {int(np.sum(out <= 0))} cells of window t={w.get('t')}")
    return out


def shadow_desc(w, mu, ok, mdl=None):
    'Window descriptors of the term: share of the fitted model intensity and count of fitted cells with mu <= 0.'
    a = shadow_term(w, mdl)
    if a is None:
        return {}
    a = np.broadcast_to(a, mu.shape)
    return dict(bg_add_share=float(np.sum(np.where(ok, a, 0.0)) / max(float(np.sum(np.where(ok, mu, 0.0))), 1e-300)),
                shadow_nonpos=int(np.sum(ok & (mu <= 0))), shadow_aB=SHADOW_AB)


def smooth3(Y, E):
    from scipy.ndimage import uniform_filter
    Ys = uniform_filter(Y, size=(1, 3, 3), mode="nearest") * 9
    Es = uniform_filter(E, size=(1, 3, 3), mode="nearest") * 9
    ok = (Es > 0) & (E > 0)
    rate = np.where(ok, Ys / np.maximum(Es, 1e-12), 0.0)
    var = np.where(ok, Ys / np.maximum(Es, 1e-12) ** 2, 0.0)
    return rate, var


class Scan:
    def __init__(self, label, grid=None):
        sys.path.insert(0, str(HERE))
        from tio2_explore import onset
        self.label = label
        self.cfg = SCANS[label]
        if AI_OVERRIDE:
            self.cfg = dict(self.cfg, ai=float(AI_OVERRIDE))
        self.metal, self.sub = self.cfg["metal"], self.cfg["sub"]
        self.grid = grid or scan_grid()
        self.q, self.af = self.grid.q, self.grid.native_af
        self.pixmask = pixmask_of(label)
        self.res = Result(IAMP.DATA / label, self.grid)
        lam = float(self.cfg.get("lam", LAM_12))
        ev = self.res.geometry.get("energy_ev")
        if ev and abs(1.239841984 / (float(ev) / 1000.0) - lam) > 1e-6:
            raise SystemExit(f"{label}: cache binned at {ev} eV, SCANS lam {lam} nm")
        self.lam_cache = lam
        if abs(lam - LAM) > 1e-9 and not LAM_REMAP:
            raise SystemExit(f"{label}: SCANS lam {lam} nm, model LAM {LAM} nm (run with IAM_LAM_NM={lam!r})")
        if self.metal == "Au" and abs(LAM - LAM_12) > 1e-9 and OPTICS == "const":
            raise SystemExit(f"{label}: the Au optics and L fluorescence are 12 keV values; IAM_LAM_NM={LAM:g} refused "
                             f"(IAM_OPTICS=table: tabulated optics at any energy)")
        if self.metal == "Au" and abs(FL_KEV - 9.70) > 1e-9:
            raise SystemExit(f"{label}: Au L fluorescence needs IAM_FL_KEV unset (9.70), got {FL_KEV:g}")
        if self.metal == "Cu" and ((abs(LAM / LAM_CU - 1) > CU_LAM_TOL and OPTICS == "const") or abs(FL_KEV - FL_KEV_CU) > 1e-9):
            raise SystemExit(f"{label}: the Cu optics are lambda {LAM_CU} nm values (used within {CU_LAM_TOL:.0%}) and "
                             f"its K fluorescence needs IAM_FL_KEV={FL_KEV_CU}; got IAM_LAM_NM={LAM:g}, IAM_FL_KEV={FL_KEV:g}")
        if self.cfg.get("pre_invalid") and os.environ.get("IAM_ALLOW_INVALID_PRE") != "1":
            raise SystemExit(f"{label}: {self.cfg['pre_invalid']} (IAM_ALLOW_INVALID_PRE=1 for data-only use)")
        self.t_dep, _, _ = onset(self.res, self.grid)
        self.t_dep_toolkit = self.t_dep
        if os.environ.get("IAM_ONSET", "toolkit") == "override" and "t_dep" in self.cfg:
            self.t_dep = float(self.cfg["t_dep"])
        no_onset = self.t_dep is None and "t_dep" in self.cfg
        if no_onset or (self.cfg.get("pre_ref") and "t_dep" in self.cfg):
            self.t_dep = float(self.cfg["t_dep"])
        self.rate_h = 0.6 * self.cfg["qcm"]
        self.clock = frame_clock(self.res.rows)
        try:
            T, X = dat_motion(self.cfg["growth"])
            self.x_of_bin = np.interp(self.clock, T, X)
        except FileNotFoundError:
            self.x_of_bin = np.zeros(len(self.clock))
        prof = yscan_profile(label)
        self.prof = prof
        cfg_onset = os.environ.get("IAM_ONSET", "toolkit") == "override" or no_onset
        pre_end = float(self.cfg.get("pre_end", self.t_dep - 0.05)) if cfg_onset else self.t_dep - 0.05
        pre_start = float(self.cfg.get("pre_start", 0.3)) if cfg_onset else 0.3
        if os.environ.get("IAM_PRE_START"):
            pre_start = float(os.environ["IAM_PRE_START"])
        if os.environ.get("IAM_PRE_END"):
            pre_end = float(os.environ["IAM_PRE_END"])
        self.pre_window = (pre_start, pre_end)
        pre = (self.res.time >= pre_start) & (self.res.time < pre_end)
        self.pre_idx = np.where(pre)[0]
        Yp, Ep = self.res.Y[pre].sum(0).astype(float), self.res.E[pre].sum(0).astype(float)
        self.I_pre, self.V_pre = smooth3(Yp, Ep)
        self.c_pre = self._cmean(self.pre_idx)
        if self.cfg.get("pre_ref"):
            ref = Scan(self.cfg["pre_ref"])
            self.I_pre, self.V_pre = ref.I_pre, ref.V_pre
            self.pre_window = (f"pre_ref {self.cfg['pre_ref']}",) + tuple(ref.pre_window)
            self.pre_idx = np.array([], dtype=int)
            self.c_pre = self._cmean(np.where(self.res.time < self.res.time.min() + 2.0)[0])
        if LAM_REMAP and abs(lam - LAM) > 1e-9 and not self.cfg.get("pre_ref"):

            self.I_pre, self.V_pre = af_remap(self.I_pre, self.af, lam), af_remap(self.V_pre, self.af, lam)
        self.shadow = None
        if SHADOW_BG != "0":
            import shadow_rows as SR
            self.shadow = SR.Shadow(label, self.res)

    def cfun(self, x):
        if self.prof is None:
            return np.ones_like(np.asarray(x, float))
        return np.interp(x, self.prof[0], self.prof[1])

    def _cmean(self, idx):
        w = self.res.E[idx].sum((1, 2, 3))
        return float(np.sum(w * self.cfun(self.x_of_bin[idx])) / max(w.sum(), 1e-12))

    def window(self, t, half=None):
        if half is None:
            half = float(os.environ.get("IAM_HALF", "0.5"))
        if t - half < float(self.cfg.get("valid_from", -1e9)):
            raise SystemExit(f"{self.label}: window {t:g} s starts before valid_from {self.cfg['valid_from']} s")
        m = (self.res.time >= t - half) & (self.res.time < t + half)
        idx = np.where(m)[0]
        Y, E = self.res.Y[idx].sum(0).astype(float), self.res.E[idx].sum(0).astype(float)
        ew = self.res.E[idx].sum((1, 2, 3))
        tm = float(np.sum(self.res.time[idx] * ew) / ew.sum())
        ok = (E > 0) & (self.I_pre > 0) & (self.af >= 0.03)[None, :, None]
        dep = max(tm - self.t_dep, 0.0)
        if "t_sput" in self.cfg:
            dep = min(dep, float(self.cfg["t_sput"]))
        h_m = self.rate_h * dep
        if os.environ.get("IAM_KAPPA"):
            dep_j = max(tm - self.t_dep - float(os.environ.get("IAM_DT", "0")), 0.05)
            if "t_sput" in self.cfg:
                dep_j = min(dep_j, float(self.cfg["t_sput"]))
            h_m = float(os.environ["IAM_KAPPA"]) * self.rate_h * dep_j
        out = dict(t=t, t_mean=tm, Y=Y, E=E, mask=ok, cfac=self._cmean(idx) / self.c_pre, h_m=h_m)
        if self.shadow is not None:
            out.update(self.shadow.window_terms(idx, self.pre_idx, SHADOW_BG, self.q, out["cfac"], SHADOW_AB))
        return out


TIO2_STACK = dict(fill=0.5, width=75.0, nslice=10)


TIO2_NUM = os.environ.get("IAM_TIO2_NUM", "0") == "1"
TIO2_NSLICE = int(os.environ.get("IAM_TIO2_NSLICE", "10"))
TIO2_DZDEEP = float(os.environ.get("IAM_TIO2_DZDEEP", "20"))
TIO2_TREF = (2 * TIO2_STACK["width"] / 10, 20.0)
R_PARTICLE = 75.0


def tio2_calibration(label):
    'Per-scan pre-state calibration of the TiO2 stack (fill, zc, wz, fbot) from the json named by IAM_TIO2_SUBW'
    path = os.environ.get("IAM_TIO2_SUBW")
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = Path(__file__).resolve().parent / path
    cal = json.loads(p.read_text(encoding="utf-8")).get("scans", {}).get(label)
    return None if cal is None else cal["adopted"]


def tio2_layers(fill=0.5, width=75.0, nslice=10, fl=False):
    'Graded top of a semi-infinite porous TiO2 layer: fill rises as tanh over 2 x width (slices), below: fill.'
    z = np.linspace(0, 2 * width, nslice + 1)
    mid = 0.5 * (z[:-1] + z[1:])
    f = fill * 0.5 * (1 + np.tanh((mid - width) / (0.4 * width)))
    dt = dc("TiO2", fl=fl)
    return [(fi * dt, 2 * width / nslice) for fi in f], f


class Model:
    def __init__(self, scan: Scan, f1_au=AU_F1_12, eps_fl=1.0, theta_max=90.0, ntheta=5, nphi=6, coarse=(4, 2),
                 nu_sub=1.0, nu_isl=0.0, sig_af=0.0, interf="da"):
        if SHADOW_BG != "0" and type(self).__module__ not in ("iam", "iam2", "iam4"):
            raise NotImplementedError(f"IAM_SHADOW_BG={SHADOW_BG}: {type(self).__name__} does not add the term")
        self.s = scan
        self.f1, self.eps_fl, self.theta_max = f1_au, eps_fl, theta_max


        self.nu_sub, self.nu_isl = nu_sub, nu_isl
        self.interf = interf
        self.af, self.q, self.ai = scan.af, scan.q, float(scan.cfg.get("ai", 0.40))
        self.set_smear(sig_af)
        self.a_all = np.r_[self.ai, self.af]
        th, wt = np.polynomial.legendre.leggauss(ntheta)
        tm = np.radians(theta_max)
        self.th = 0.5 * tm * (th + 1)
        w = np.sin(self.th) * np.cos(self.th) * 0.5 * tm * wt
        self.wth = w / w.sum()
        ph, wp = np.polynomial.legendre.leggauss(nphi)
        self.ph = 0.5 * np.pi * (ph + 1); self.wph = 0.5 * wp
        self.rho_top = rho_e(scan.sub) if scan.sub in POLY else rho_e("Si")
        if scan.sub == "TiO2":
            cal = tio2_calibration(scan.label)
            self.set_tio2_stack(cal["fill"] if cal else TIO2_STACK["fill"])
            if cal:
                self.tio2_subw = (float(cal["zc"]), float(cal["wz"]), float(cal["fbot"]))
        elif scan.sub in POLY:
            self.set_poly_t(scan.cfg.get("poly_t"))
        else:
            self.sub_layers, self.fills, self.sub_layers_fl = [], np.array([]), []
            self.d_sub, self.d_sub_fl = dc("Si"), dc("Si", fl=True)
        self.ia = np.arange(0, len(self.af), coarse[0]); self.iq = np.arange(0, len(self.q), coarse[1])
        if self.ia[-1] != len(self.af) - 1:
            self.ia = np.r_[self.ia, len(self.af) - 1]
        if self.iq[-1] != len(self.q) - 1:
            self.iq = np.r_[self.iq, len(self.q) - 1]
        self._pre = None

    tio2_subw = None

    def set_tio2_stack(self, fill):
        'Graded porous TiO2 stack with TiO2 volume fraction fill (default TIO2_STACK; per scan from the pre-state'
        self.tio2_fill = float(fill)
        if not TIO2_NUM and (TIO2_NSLICE, TIO2_DZDEEP) != (10, 20.0) and os.environ.get("TIO2_PREREQ_FORCE", "0") != "1":
            raise SystemExit("IAM_TIO2_NSLICE / IAM_TIO2_DZDEEP need IAM_TIO2_NUM=1 (legacy per-row weights scale with the grid)")
        if os.environ.get("IAM_TIO2_DEEP", "0") == "1":
            r_ = float(os.environ.get("IAM_TIO2_DEEP_NM", "200")) / TIO2_DZDEEP
            if abs(r_ - round(r_)) > 1e-9:
                raise SystemExit("IAM_TIO2_DEEP_NM must be a multiple of IAM_TIO2_DZDEEP")
        st = dict(TIO2_STACK, fill=self.tio2_fill, nslice=TIO2_NSLICE)
        self.sub_layers, self.fills = tio2_layers(**st)
        self.sub_layers_fl, _ = tio2_layers(**st, fl=True)
        self.d_sub, self.d_sub_fl = self.tio2_fill * dc("TiO2"), self.tio2_fill * dc("TiO2", fl=True)
        self._region = np.zeros(len(self.sub_layers), int)
        if os.environ.get("IAM_TIO2_DEEP", "0") == "1":
            fl0, dz = self.tio2_fill, TIO2_DZDEEP
            nd = max(int(round(float(os.environ.get("IAM_TIO2_DEEP_NM", "200")) / dz)), 1)
            self.sub_layers = self.sub_layers + [(fl0 * dc("TiO2"), dz)] * nd
            self.sub_layers_fl = self.sub_layers_fl + [(fl0 * dc("TiO2", fl=True), dz)] * nd
            self.fills = np.r_[self.fills, np.full(nd, fl0)]
            self._region = np.r_[self._region, np.ones(nd, int)]
        self._pre = None

    def tio2_sub_weights(self):
        'Depth weights of the TiO2 scatterers in the substrate term: one per graded / deep slice and one for the'
        if self.tio2_subw is None:
            if TIO2_NUM:
                t = np.array([tt for _, tt in self.sub_layers])
                tref = np.where(self._region == 0, TIO2_TREF[0], TIO2_TREF[1])
                return np.r_[self.fills * t / tref, self.tio2_fill]
            return np.r_[self.fills, self.tio2_fill]
        zc, wz, fbot = self.tio2_subw
        t = np.array([tt for _, tt in self.sub_layers])
        zm = np.cumsum(t) - 0.5 * t
        g = self.fills * t * np.exp(-0.5 * ((zm - zc) / max(wz, 1e-3)) ** 2)
        g = (1.0 - fbot) * g / max(g.sum(), 1e-300)
        return np.r_[g, fbot]

    def set_poly_t(self, t):
        'Polymer film of thickness t (nm) on Si; t None or <= 0: a bulk polymer substrate.'
        self.poly_t = float(t) if t else None
        if self.poly_t:
            self.sub_layers, self.sub_layers_fl = [(dc(self.s.sub), self.poly_t)], [(dc(self.s.sub, fl=True), self.poly_t)]
            self.d_sub, self.d_sub_fl = dc("Si"), dc("Si", fl=True)
        else:
            self.sub_layers, self.sub_layers_fl = [], []
            self.d_sub, self.d_sub_fl = dc(self.s.sub), dc(self.s.sub, fl=True)
        self.fills = np.array([])
        self._pre = None

    def flat_layers(self):
        'Flat layers between the metal and the semi-infinite substrate (a polymer film; none on Si / TiO2 here).'
        return self.sub_layers if self.s.sub in POLY else []


    smear_edge = os.environ.get("IAM_SMEAR_EDGE", "legacy")

    def smear_kernel(self, sig):
        'alpha_f smear matrix (n_af, n_af) of rms sig (deg) for the IAM_SMEAR_EDGE rule; None for sig <= 0.'
        sig = float(sig)
        if sig > 0 and self.smear_edge == "zero":
            from scipy.special import ndtr
            af = np.asarray(self.af, float)
            e = np.r_[af[0] - 0.5 * (af[1] - af[0]), 0.5 * (af[1:] + af[:-1]), af[-1] + 0.5 * (af[-1] - af[-2])]
            cdf = ndtr((e[None, :] - af[:, None]) / sig)
            k = np.diff(cdf, axis=1)
            k[:, -1] += 1.0 - cdf[:, -1]
            return k
        if sig > 0:
            d = self.af[:, None] - self.af[None, :]
            k = np.exp(-0.5 * (d / sig) ** 2) * np.gradient(self.af)[None, :]
            return k / k.sum(1, keepdims=True)
        return None

    def set_smear(self, sig_af):
        'Gaussian alpha_f smear (deg) of the computed terms (metal term and field-factor ratio G); the measured'
        self.sig_af = float(sig_af)
        self.Kaf = self.smear_kernel(self.sig_af)

    def smear(self, A):
        return A if self.Kaf is None else self.Kaf @ A

    def smear_ratio(self, G):
        'Smear of a field-factor ratio (1-D), normalised by the smear of a constant: G = 1 (no metal) leaves the'
        if self.Kaf is None:
            return G
        return (self.Kaf @ G) / np.maximum(self.Kaf.sum(1), 1e-12)

    tio2_lam = None
    tio2_bury = None

    def tio2_metal(self, h_m):
        'Metal fraction per graded TiO2 slice: line-of-sight deposit on the exposed upper hemispheres of particles of'
        t = np.array([tt for _, tt in self.sub_layers])
        zmid = np.cumsum(t) - 0.5 * t
        if self.tio2_bury is not None:
            from scipy.special import erf
            z0, wb = self.tio2_bury
            zb = np.r_[0.0, np.cumsum(t)]
            c = 0.5 * (1 + erf((zb - z0) / (np.sqrt(2) * max(wb, 1e-3))))
            w = np.diff(c)
            zpk = min(max(z0, 0.0), zb[-1])
            self._rho_peak = h_m / max(w.sum(), 1e-12) * np.exp(-0.5 * ((zpk - z0) / max(wb, 1e-3)) ** 2) \
                / (np.sqrt(2 * np.pi) * max(wb, 1e-3))
            return h_m * w / max(w.sum(), 1e-12) / t
        if self.tio2_lam:
            w = np.exp(-zmid / self.tio2_lam) * t
            self._rho_peak = h_m / max(w.sum(), 1e-12)
            return h_m * w / max(w.sum(), 1e-12) / t
        w = np.clip(1 - zmid / R_PARTICLE, 0, None) * t
        self._rho_peak = h_m / max(w.sum(), 1e-12)
        return h_m * w / max(w.sum(), 1e-12) / t

    cap = None

    def tio2_fields(self, h_m, metal=True, k=K0, fl=False):
        '|E|^2 at the graded slices (n_slice, 1 + n_af) and the substrate-scatterer factor for a metal load. With'
        phm = self.tio2_metal(h_m) if metal else np.zeros(len(self.sub_layers))
        dm = dc(self.s.metal, fl=fl, f1_au=self.f1)
        base = self.sub_layers_fl if fl else self.sub_layers
        layers = [(d + self.nu_sub * p * dm, t) for (d, t), p in zip(base, phm)]
        if metal and self.cap is not None and self.cap[1] > 0:
            layers = [(self.nu_sub * self.cap[0] * dm, self.cap[1])] + layers

        a = self.a_all if not fl else np.maximum(getattr(self, "af_c", self.af), 1e-4)
        I = layer_mid_int(a, layers, self.d_sub_fl if fl else self.d_sub, k=k)
        return I, phm

    def fields(self, slab):
        '|E|^2 at the slab slices (len(slab), 1 + n_af) and the substrate-scatterer factor (n_af).'
        layers = slab + self.sub_layers
        I = layer_mid_int(self.a_all, layers, self.d_sub)
        Ise = I[: len(slab)]
        if self.s.sub == "TiO2":
            It = np.r_[I[len(slab): len(slab) + len(self.sub_layers)], I[-1:]]
            fl = self.tio2_sub_weights()
            G = np.sum(fl[:, None] * It[:, :1] * It[:, 1:], axis=0)
        elif self.s.sub in POLY and self.sub_layers:
            Ip = I[len(slab)]
            G = Ip[0] * Ip[1:]
        else:
            G = I[-1, 0] * I[-1, 1:]
        return Ise, G

    def pre_factor(self):
        if self._pre is None:
            if self.s.sub == "TiO2":
                I, _ = self.tio2_fields(0.0, metal=False)
                fl = self.tio2_sub_weights()
                self._pre = np.sum(fl[:, None] * I[:, :1] * I[:, 1:], axis=0)
            else:
                self._pre = self.fields([])[1]
        return self._pre

    def islands_si(self, m: Morph, slab):
        H = m.Hmax
        layer = [(sum(d * t for d, t in slab) / max(sum(t for _, t in slab), 1e-9), H)]
        Ti, Ri, kzi = base_amps(np.array([self.ai]), layer, self.d_sub, 1)
        Tf, Rf, kzf = base_amps(self.af, layer, self.d_sub, 1)
        kzi, Ti, Ri = kzi[0], Ti[0], Ri[0]
        terms = [(kzf + kzi, Ti * Tf), (kzf - kzi, Ri * Tf), (-kzf + kzi, Ti * Rf), (-kzf - kzi, Ri * Rf)]
        F1 = 0; F2 = 0; L = 0
        Rm = float(np.sum(m.wj * m.Rj))
        for Rj, Hj, wj in zip(m.Rj, m.Hj, m.wj):
            Fj = 0
            for Qz, amp in terms:
                Fj = Fj + amp[:, None] * ff_sep(self.q, Qz, Rj, Hj)
            F1 = F1 + wj * Fj; F2 = F2 + wj * np.abs(Fj) ** 2
            if self.interf == "lma":
                L = L + wj * np.abs(Fj) ** 2 * s_para(self.q, lma_D(m.D, Rj, Rm), m.omega)[None, :]
        rho = rho_e(self.s.metal, self.f1)
        if self.interf == "lma":
            return m.n * RE ** 2 * abs(rho) ** 2 * L
        S = s_para(self.q, m.D, m.omega)[None, :]
        return m.n * RE ** 2 * abs(rho) ** 2 * (F2 - np.abs(F1) ** 2 + np.abs(F1) ** 2 * S)

    def born_tio2(self, m: Morph):
        'Orientation-averaged Born (no field factors) on the coarse grid, interpolated to the full grid.'
        af, q = self.af[self.ia], self.q[self.iq]
        qz = K0 * (np.sin(np.radians(self.ai)) + np.sin(np.radians(af)))
        out = 0
        for th, wt in zip(self.th, self.wth):
            ct, st = np.cos(th), np.sin(th)
            D_loc = m.D / np.sqrt(max(ct, 1e-3))
            for ph, wp in zip(self.ph, self.wph):
                qzl = q[None, :] * st * np.sin(ph) + qz[:, None] * ct
                qpl = np.sqrt(np.maximum(q[None, :] ** 2 + qz[:, None] ** 2 - qzl ** 2, 0.0))
                F1 = 0; F2 = 0; L = 0
                Rm = float(np.sum(m.wj * m.Rj))
                for Rj, Hj, wj in zip(m.Rj, m.Hj, m.wj):
                    Fj = ff_arr(qpl, qzl, Rj, Hj)
                    F1 = F1 + wj * Fj; F2 = F2 + wj * np.abs(Fj) ** 2
                    if self.interf == "lma":
                        L = L + wj * np.abs(Fj) ** 2 * s_para(qpl, lma_D(D_loc, Rj, Rm), m.omega)
                if self.interf == "lma":
                    out = out + wt * wp * L
                else:
                    out = out + wt * wp * (F2 - np.abs(F1) ** 2 + np.abs(F1) ** 2 * s_para(qpl, D_loc, m.omega))

        lo = np.log(np.maximum(out, 1e-300))
        A1 = np.array([np.interp(self.af, af, lo[:, j]) for j in range(lo.shape[1])]).T
        A2 = np.array([np.interp(np.log(self.q), np.log(q), A1[i]) for i in range(A1.shape[0])])
        return m.n * np.exp(A2)

    def r_coat(self, m: Morph):
        qz = K0 * (np.sin(np.radians(self.ai)) + np.sin(np.radians(self.af)))
        qa = np.sqrt(qz[:, None] ** 2 + self.q[None, :] ** 2)
        cpsi = qz[:, None] / qa
        rT, rM = rho_e("TiO2"), rho_e(self.s.metal, self.f1)
        zs = np.linspace(0, m.Hmax, 16)
        dz = zs[1] - zs[0]
        phi = m.phi_profile(zs)
        Phi = (np.exp(1j * qa[..., None] * zs) @ (phi * dz)) * cpsi
        val = np.abs(1 + 1j * qa * (rM / rT) * Phi) ** 2
        return np.where(np.degrees(np.arccos(np.clip(cpsi, 0, 1))) <= self.theta_max, val, 1.0)

    def fluor(self, m: Morph, slab, phi, Ise, rows_fl=None, tail_fl=None):
        'rows_fl / tail_fl (underlayer.buried_fluor): explicit fluorescence-energy decrements of the slab rows and the'
        if self.s.metal not in ("Au", "Cu") or m.h_m <= 0 or not fluorescence_on(self.s.metal):
            return np.zeros(len(self.af))
        if SMOOTH:
            wt = np.asarray(phi, float) * np.array([t for _, t in slab][: len(phi)], float)
            wt = wt / max(wt.sum(), 1e-12)
        else:
            wt = phi / max(phi.sum(), 1e-12)
        Ei = float(np.sum(wt * Ise[:, 0]))
        dmf = dc(self.s.metal, fl=True)
        slab_fl = [(p * self.nu_sub * dmf, t) for p, (d, t) in zip(phi, slab)] if rows_fl is None else list(rows_fl)
        tail = self.sub_layers_fl if tail_fl is None else list(tail_fl)
        If = layer_mid_int(self.af, slab_fl + tail, self.d_sub_fl, k=K_FL)[: len(slab)]
        Ef = np.sum(wt[:, None] * If, axis=0)
        if self.s.metal == "Cu":
            return self.eps_fl * EFF_RATIO_CU * P_K_CU * MU_CU * m.h_m * Ei * OMEGA_K_CU / (4 * np.pi) * Ef
        return self.eps_fl * EFF_RATIO * P_L3 * MU_AU_12 * m.h_m * Ei * OMEGA_L3 / (4 * np.pi) * Ef

    def r_rep_flat(self, m: Morph):
        'Replication of the substrate roughness by the metal on a flat substrate. Islands sit on the rough surface'
        qz = K0 * (np.sin(np.radians(self.ai)) + np.sin(np.radians(self.af)))
        rS, rM = self.rho_top, rho_e(self.s.metal, self.f1)
        Fm = 0
        for Rj, Hj, wj in zip(m.Rj, m.Hj, m.wj):
            Fm = Fm + wj * ff_sep(self.q, qz, Rj, Hj)
        Rz = np.abs(1 + 1j * qz[:, None] * (rM / rS) * m.n * Fm) ** 2
        return Rz - 1

    def predict(self, w, m: Morph, K, g=0.0, parts=False):
        rho2 = RE ** 2 * abs(rho_e(self.s.metal, self.f1)) ** 2
        if self.s.sub == "TiO2":
            I, phm = self.tio2_fields(m.h_m)
            fl_w = self.tio2_sub_weights()
            Gr = np.sum(fl_w[:, None] * I[:, :1] * I[:, 1:], axis=0) / self.pre_factor()
            sub = w["cfac"] * Gr[None, :, None] * self.s.I_pre
            if g > 0:
                sub = sub * (1 - g + g * self.r_coat(m))[None]
            wt = phm / max(phm.sum(), 1e-12)
            Is = I[: len(phm)]
            Ei, Ef = float(np.sum(wt * Is[:, 0])), np.sum(wt[:, None] * Is[:, 1:], axis=0)
            isl = Ei * Ef[:, None] * rho2 * self.born_tio2(m)
            fl = np.zeros(len(self.af))
            if self.s.metal == "Au" and m.h_m > 0:
                If, _ = self.tio2_fields(m.h_m, k=K_FL, fl=True)
                Eff = np.sum(wt[:, None] * If[: len(phm)], axis=0)
                fl = self.eps_fl * EFF_RATIO * P_L3 * MU_AU_12 * m.h_m * Ei * OMEGA_L3 / (4 * np.pi) * Eff
            phi = phm
        else:
            slab0, phi = m.slab(self.s.metal, f1_au=self.f1)
            slab = [(self.nu_sub * d, t) for d, t in slab0]
            Ise, G = self.fields(slab)
            Gr = G / self.pre_factor()
            sub = w["cfac"] * Gr[None, :, None] * self.s.I_pre
            if g > 0:
                sub = sub * (1 + g * self.r_rep_flat(m))[None]
            isl = self.islands_si(m, [(self.nu_isl * d, t) for d, t in slab0])
            fl = self.fluor(m, slab, phi, Ise)
        metal = self.smear(isl + fl[:, None])
        if self.Kaf is not None:
            sub = sub * (self.smear_ratio(Gr) / np.maximum(Gr, 1e-12))[None, :, None]
        mu = shadow_add(w, sub + K * metal[None])
        if parts:
            return mu, dict(sub=sub, metal=metal, isl=isl, fl=fl, G=Gr, coverage=m.coverage, D=m.D, n=m.n,
                            phi=np.asarray(phi).tolist())
        return mu, sub, metal


def dev5(w, mu, Vsub, f=0.05):
    E, Y, ok = w["E"], w["Y"], w["mask"]
    m = np.maximum(mu * E, 1e-6)
    vpre = Vsub * E * E
    den = 1 + f * f * m + vpre / m
    d = np.maximum(2 * (xlogy(Y, Y / m) - Y + m) / den, 0.0)
    return np.where(ok, d, 0.0)


def best_K(w, sub, metal, Vsub, K0_=None, iters=30):
    'Profile the deviance over K >= 0 (1D, golden section on log K around the weighted LS estimate).'
    E, Y, ok = w["E"], w["Y"], w["mask"]
    Mfull = np.broadcast_to(metal[None], sub.shape)
    resid = np.where(ok, Y / np.maximum(E, 1e-12) - sub, 0.0)
    wt = np.where(ok, E / np.maximum(sub + 1e-3, 1e-3), 0.0)
    num, den = np.sum(wt * Mfull * resid), np.sum(wt * Mfull * Mfull)
    k0 = max(num / max(den, 1e-300), 1e-6) if K0_ is None else K0_

    def f(lk):
        return float(dev5(w, sub + np.exp(lk) * Mfull, Vsub).sum())
    a, b = np.log(k0) - 3, np.log(k0) + 3
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = f(d)
    lk = 0.5 * (a + b)
    return float(np.exp(lk)), f(lk)


__all__ = ["Scan", "Model", "Morph", "dev5", "best_K", "stack", "layer_mid_int", "ff_sep", "dc", "rho_e"]
