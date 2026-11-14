'TiO2 buried-metal terms beyond the production Born approximation (results/iam/tio2_contrast_preregistration.json).'
from __future__ import annotations

import math
import os

import numpy as np
from scipy.special import j1, jv

import iam as M

AZ = os.environ.get("IAM_TIO2_AZ", "half")
_OR = os.environ.get("IAM_TIO2_ORIENT", "").strip()
ORIENT = tuple(int(v) for v in _OR.split(",")) if _OR else None
PHI = os.environ.get("IAM_TIO2_PHI", "uni")
DWBA = int(os.environ.get("IAM_TIO2_DWBA", "0"))
NB = int(os.environ.get("IAM_TIO2_DWBA_NB", "8"))
NET = int(os.environ.get("IAM_TIO2_DWBA_NET", "0"))
JINC = os.environ.get("IAM_TIO2_JINC", "exact")
CONTRAST = os.environ.get("IAM_TIO2_CONTRAST", "vac")
TESTHOOK = os.environ.get("IAM_TIO2_TESTHOOK", "0") == "1"
QZVAC = os.environ.get("IAM_TIO2_QZVAC", "0") == "1"
FFTEST = os.environ.get("IAM_TIO2_FFTEST", "")
ROWS = os.environ.get("IAM_TIO2_DWBA_ROWS", "coarse")
L5_SCALE = float(os.environ.get("IAM_TIO2_L5_SCALE", "1"))


ORIENT_MODE = os.environ.get("IAM_TIO2_ORIENT_MODE", "plain")
_ORS = os.environ.get("IAM_TIO2_ORIENT_S", "32,20").strip()
ORIENT_S = tuple(int(v) for v in _ORS.split(","))

ISLAND = AZ == "full" or ORIENT is not None or DWBA != 0
ACTIVE = ISLAND or NET != 0 or CONTRAST != "vac"


def check_config(tio2_fix, tio2_num):
    'Guards (registration implementation.guards); called once by iam2 at import.'
    if AZ not in ("half", "full") or PHI not in ("uni", "gl") or DWBA not in (0, 1, 4, 5) or NET not in (0, 1, 2) \
            or JINC not in ("exact", "re") or CONTRAST not in ("vac", "em", "pore") or ROWS not in ("coarse", "full", "step12"):
        raise SystemExit("IAM_TIO2_AZ/PHI/DWBA/DWBA_NET/JINC/CONTRAST/DWBA_ROWS: unknown value")
    if not L5_SCALE > 0:
        raise SystemExit("IAM_TIO2_L5_SCALE must be > 0")
    if ORIENT is not None and (len(ORIENT) != 2 or min(ORIENT) < 1):
        raise SystemExit("IAM_TIO2_ORIENT must be 'ntheta,nphi'")
    if ORIENT_MODE not in ("plain", "nested") or len(ORIENT_S) != 2 or min(ORIENT_S) < 1:
        raise SystemExit("IAM_TIO2_ORIENT_MODE plain|nested, IAM_TIO2_ORIENT_S 'ntheta,nphi'")
    if ORIENT_MODE == "nested" and (AZ != "full" or ORIENT is None or min(ORIENT) < 2):
        raise SystemExit("IAM_TIO2_ORIENT_MODE=nested needs IAM_TIO2_AZ=full and IAM_TIO2_ORIENT 'nt,ns' (>= 2)")
    if QZVAC and not TESTHOOK:
        raise SystemExit("IAM_TIO2_QZVAC is a test hook (needs IAM_TIO2_TESTHOOK=1)")
    if FFTEST not in ("", "sphere", "hemi0") or (FFTEST and not TESTHOOK):
        raise SystemExit("IAM_TIO2_FFTEST sphere|hemi0 is a test hook (needs IAM_TIO2_TESTHOOK=1)")
    if (DWBA != 0 or NET != 0) and not TESTHOOK:
        if not (tio2_fix and tio2_num and AZ == "full"):
            raise SystemExit("IAM_TIO2_DWBA / IAM_TIO2_DWBA_NET need IAM_TIO2_FIX=1, IAM_TIO2_NUM=1 and IAM_TIO2_AZ=full")


def assert_fit_allowed():
    'Fit entry points (iam_fit_t, refit_branches): no sensitivity contrast, no test hooks.'
    if CONTRAST != "vac":
        raise SystemExit("IAM_TIO2_CONTRAST != vac is for fixed-parameter evaluation only")
    if TESTHOOK or QZVAC or FFTEST:
        raise SystemExit("IAM_TIO2_TESTHOOK / IAM_TIO2_QZVAC / IAM_TIO2_FFTEST are for the gates and selftests only")


_NS = 60
_COEF = 0.5 * np.array([(-0.25) ** k / (math.factorial(k) * math.factorial(k + 1)) for k in range(_NS)])
_XS = 225.0


def _hankel_j1(z):
    mu = 4.0
    w = 8 * z
    P = 1 - (mu - 1) * (mu - 9) / (2 * w ** 2) + (mu - 1) * (mu - 9) * (mu - 25) * (mu - 49) / (24 * w ** 4)
    Q = (mu - 1) / w - (mu - 1) * (mu - 9) * (mu - 25) / (6 * w ** 3) \
        + (mu - 1) * (mu - 9) * (mu - 25) * (mu - 49) * (mu - 81) / (120 * w ** 5)
    chi = z - 0.75 * np.pi
    return np.sqrt(2 / (np.pi * z)) * (P * np.cos(chi) - Q * np.sin(chi))


def jinc_complex(X, chunk=1 << 15):
    'J1(sqrt X) / sqrt X for complex X (any shape; even in sqrt X, branch free).'
    X = np.asarray(X, complex)
    flat = X.ravel()
    out = np.empty_like(flat)
    big = (flat.real * flat.real + flat.imag * flat.imag) > _XS * _XS
    nbig = int(big.sum())
    if nbig:
        z = np.sqrt(flat[big])
        z = np.where(z.real < 0, -z, z)
        out[big] = _hankel_j1(z) / z
    src = flat if nbig == 0 else flat[~big]
    res = np.empty_like(src)
    for i in range(0, src.size, chunk):
        Xi = src[i:i + chunk]
        acc = np.full(Xi.shape, _COEF[-1], complex)
        for c in _COEF[-2::-1]:
            acc *= Xi
            acc += c
        res[i:i + chunk] = acc
    if nbig == 0:
        out = res
    else:
        out[~big] = res
    return out.reshape(X.shape)


def jinc_real(x):
    x = np.asarray(x, float)
    out = np.full(x.shape, 0.5)
    m = np.abs(x) > 1e-6
    out[m] = j1(x[m]) / x[m]
    return out


def jinc_X(X, mode=None):
    "J1(sqrt X)/sqrt X; mode 'exact' (complex), 're' (Re X only); purely real input uses scipy j1 (= production)."
    mode = JINC if mode is None else mode
    X = np.asarray(X)
    if not np.iscomplexobj(X) or not np.any(X.imag):
        return jinc_real(np.sqrt(np.maximum(np.real(X), 0.0))) + 0j
    if mode == "re":
        return jinc_real(np.sqrt(np.maximum(X.real, 0.0))) + 0j
    return jinc_complex(X)


def orientation_nodes(ntheta, nphi, theta_max, az=None, phi=None):
    'Cap tilt theta (Gauss-Legendre on [0, theta_max], weights ~ sin cos, as production) and the azimuth as signed'
    az = AZ if az is None else az
    phi = PHI if phi is None else phi
    th, wt = np.polynomial.legendre.leggauss(ntheta)
    tm = np.radians(theta_max)
    th = 0.5 * tm * (th + 1)
    w = np.sin(th) * np.cos(th) * 0.5 * tm * wt
    wth = w / w.sum()
    if az == "half":
        ph, wp = np.polynomial.legendre.leggauss(nphi)
        s, ws = np.sin(0.5 * np.pi * (ph + 1)), 0.5 * wp
    elif phi == "gl":
        ph, wp = np.polynomial.legendre.leggauss(nphi)
        s0 = np.sin(0.5 * np.pi * (ph + 1))
        s, ws = np.r_[s0, -s0], np.r_[0.25 * wp, 0.25 * wp]
    else:


        half = np.arange((nphi + 1) // 2)
        sp = np.sin(np.pi * (half + 0.5) / nphi)
        wpair = np.where(half == nphi - 1 - half, 1.0, 2.0) / (2.0 * nphi)
        s, ws = np.r_[-sp[::-1], sp], np.r_[wpair[::-1], wpair]
    if FFTEST == "hemi0":
        th, wth = np.zeros(1), np.ones(1)
    return th, wth, np.asarray(s, float), np.asarray(ws, float)


def _rows(mdl):
    'The alpha_f rows on which the island term is evaluated (IAM_TIO2_DWBA_ROWS).'
    if ROWS == "full":
        return np.arange(len(mdl.af))
    if ROWS == "step12":
        r = np.asarray(mdl.ia)[::2]
        return r if r[-1] == mdl.ia[-1] else np.r_[r, mdl.ia[-1]]
    return mdl.ia


def _settings():
    'Module state that changes the island term (part of every cache key).'
    return (AZ, ORIENT, PHI, DWBA, NB, JINC, QZVAC, ROWS, ORIENT_MODE, ORIENT_S, L5_SCALE, FFTEST)


def model_nodes(mdl):
    'The orientation nodes the new island code uses for this model (cached with the settings that made them).'
    key = _settings() + (mdl.theta_max, len(mdl.th), len(mdl.ph))
    cached = getattr(mdl, "_tnodes", None)
    if cached is None or cached[0] != key:
        nt, npf = ORIENT if ORIENT is not None else (len(mdl.th), len(mdl.ph))
        mdl._tnodes = (key, orientation_nodes(nt, npf, mdl.theta_max))
    return mdl._tnodes[1]


def amplitudes(mdl, phm, wt, wm=None):
    'Complex up/down amplitudes at the middle of every graded slice for the metal-loaded stack (all angles), the'
    if mdl.cap is not None:
        raise SystemExit("IAM_TIO2_DWBA / DWBA_NET: cap layers are not supported")
    dm = M.dc(mdl.s.metal, f1_au=mdl.f1)
    layers = [(d + mdl.nu_sub * p * dm, t) for (d, t), p in zip(mdl.sub_layers, phm)]
    kzs, A, B, ts = M.stack(mdl.a_all, layers, mdl.d_sub)
    N = len(layers)
    k = np.array(kzs[1:N + 1])
    t = np.array(ts[1:N + 1])[:, None]
    Cd = np.array(A[1:N + 1]) * np.exp(1j * k * 0.5 * t)
    Cu = np.array(B[1:N + 1]) * np.exp(-1j * k * 0.5 * t)
    Ichk = np.abs(Cd + Cu) ** 2
    Iref = M.layer_mid_int(mdl.a_all, layers, mdl.d_sub)[:N]
    if not np.allclose(Ichk, Iref, rtol=1e-9, atol=0):
        raise AssertionError("tio2_dwba: slice amplitudes do not reproduce layer_mid_int")
    return dict(k=k, Cd=Cd, Cu=Cu, d=np.array([d for d, _ in layers]), wt=np.asarray(wt, float), t=t[:, 0],
                wm=np.asarray(wt if wm is None else wm, float),
                vac=(np.asarray(kzs[0]), np.asarray(A[0]), np.asarray(B[0])))


def channels(amp, cols):
    'C_t (4, N, n_cols) and the sign pattern of Q_t = s_i k_i + s_f k_f.'
    Di, Ui = amp["Cd"][:, :1], amp["Cu"][:, :1]
    Df, Uf = amp["Cd"][:, cols], amp["Cu"][:, cols]
    C = np.array([Di * Df, Ui * Df, Di * Uf, Ui * Uf])
    sgn = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    return C, sgn


def bins(amp, nb):
    'Slices grouped by Re(decrement) into nb bins of equal width (empty bins dropped); slice -> bin and the'
    wt, d = amp["wt"], amp["d"]
    keep = wt > 1e-14 * wt.max()
    dr = d.real
    lo, hi = dr[keep].min(), dr[keep].max()
    edges = np.linspace(lo, hi + 1e-30 + 1e-9 * abs(hi), max(int(nb), 1) + 1)
    bi = np.clip(np.searchsorted(edges, dr, side="right") - 1, 0, max(int(nb), 1) - 1)
    bi[~keep] = -1
    used = [b for b in range(max(int(nb), 1)) if np.any(bi == b)]
    dcen = np.array([np.sum(wt[bi == b] * d[bi == b]) / np.sum(wt[bi == b]) for b in used])
    remap = {b: j for j, b in enumerate(used)}
    return np.array([remap.get(b, -1) for b in bi]), dcen


_NH = len(M.GL_S) // 2
_S_LO = M.GL_S[:_NH]
assert len(M.GL_S) % 2 == 0 and np.allclose(M.GL_S[::-1], 1 - M.GL_S, atol=1e-14), "height nodes not symmetric"


def _g_part(Kt2, R, H, mode):
    'Bessel part of the hemispheroid amplitude for size nodes R (n,): (n, ..., 20) complex.'
    sh = (-1,) + (1,) * Kt2.ndim
    R = np.asarray(R, float).reshape(sh)
    X = (Kt2[None, ..., None] * (R * R)[..., None]) * M.GL_R ** 2
    return (M.GL_R ** 2 * M.GL_W) * jinc_X(X, mode)


def _ff_from_g(g, Kn, R, H):
    'F_c = 2 pi R^2 H sum_k g_k exp(i Kn H (s_k - 3/8)): (n, ...).'
    sh = (-1,) + (1,) * Kn.ndim
    R = np.asarray(R, float).reshape(sh)
    H = np.asarray(H, float).reshape(sh)
    ph = 1j * Kn[None] * H

    e_lo = np.exp(ph[..., None] * (_S_LO - 0.375))
    e_hi = np.exp(0.25 * ph)[..., None] / e_lo
    return 2 * np.pi * (R * R * H) * (np.einsum("...k,...k->...", g[..., :_NH], e_lo)
                                      + np.einsum("...k,...k->...", g[..., _NH:][..., ::-1], e_hi))


def _size_nodes(pop):
    if M.QUAD == "gh":
        return pop.Rj, pop.Hj, pop.wj
    pRj, pwj = M.ln_nodes(pop.R, pop.s, None, None, dr=M.QUAD_DR_ORIENT, rel=M.QUAD_REL_ORIENT)
    return pRj, pop.a * pRj, pwj


def _lma(mdl, qpl, Rj, Rm, D, omega, lma, eta, ct):
    from iam2 import s_hd
    if lma and mdl.interf == "hd":
        return np.array([s_hd(qpl * 2 * r_ * (1 + omega), eta * ct) for r_ in Rj])
    if lma:
        return M.s_para(qpl[None], M.lma_D(D / np.sqrt(max(ct, 1e-3)), Rj, Rm)[:, None, None], omega)
    return np.ones((len(Rj),) + qpl.shape)


def _interp_to_full(mdl, af_c, lo):
    'log intensity on (coarse rows, coarse q) -> full grid (as production).'
    if lo.shape[0] != len(mdl.af_c):
        lo = np.array([np.interp(mdl.af_c, af_c, lo[:, j]) for j in range(lo.shape[1])]).T
    return np.array([np.interp(np.log(mdl.q), np.log(mdl.q[mdl.iq]), lo[i]) for i in range(lo.shape[0])])


def born_pop(mdl, pop, n, D, omega, lma, eta=0.0):
    'Replacement of iam2.Model2.born_pop_tio2 when any island switch is set. Level 0: production Born on the new'
    key = _born_key(mdl, pop, n, D, omega, lma, eta)
    cache = getattr(mdl, "_born_cache", None)
    if cache is None:
        cache = mdl._born_cache = {}
    if key in cache:
        out = cache.pop(key)
        cache[key] = out
        return out
    if DWBA == 5:
        out = _born_level5(mdl, pop, n, D, omega, lma, eta, nested=ORIENT_MODE == "nested")
    elif ORIENT_MODE == "nested":
        out = _born_nested(mdl, pop, n, D, omega, lma, eta)
    else:
        out = _born_plain(mdl, pop, n, D, omega, lma, eta)
    out.setflags(write=False)
    if len(cache) >= 8:
        cache.pop(next(iter(cache)))
    cache[key] = out
    return out


def _born_key(mdl, pop, n, D, omega, lma, eta):
    'Everything the island term depends on (sizes, spacing, angles, and for the DWBA levels the slice amplitudes'
    import hashlib
    h = hashlib.sha1()
    h.update(repr(_settings() + (mdl.theta_max, len(mdl.th), len(mdl.ph))).encode())
    for v in (pop.R, pop.s, pop.a, n, D, omega, float(bool(lma)), eta, mdl.ai):
        h.update(np.float64(v).tobytes())
    for arr in (pop.Rj, pop.Hj, pop.wj, mdl.af_c, mdl.q):
        h.update(np.ascontiguousarray(arr, dtype=float).tobytes())
    h.update(str(mdl.interf).encode())
    if DWBA != 0:
        a = mdl._amp
        for arr in (a["Cd"], a["Cu"], a["k"], a["wt"], a["t"], np.asarray(mdl._EiEf)):
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _dwba_channels(mdl, af_c):
    'Levels 1, 4 on the rows af_c: slice -> bin map, number of bins, channel wave-vector sums Q12 and number of'
    amp = mdl._amp
    bj, dcen = bins(amp, NB)
    nb = len(dcen)
    if QZVAC:
        kib = np.full((nb, 1), M.K0 * np.sin(np.radians(mdl.ai)), complex)
        kfb = np.broadcast_to(M.K0 * np.sin(np.radians(np.maximum(af_c, 0.0)))[None, :], (nb, len(af_c))) + 0j
    else:
        kb = M.kz(np.r_[mdl.ai, np.maximum(af_c, 1e-4)][None, :], dcen[:, None], M.K0)
        kib, kfb = kb[:, :1], kb[:, 1:]
    if DWBA == 1 or QZVAC:
        return amp, bj, nb, (kib + kfb)[:, None, :], 1
    return amp, bj, nb, np.stack([kib + kfb, -kib + kfb], axis=1), 4


def _born_plain(mdl, pop, n, D, omega, lma, eta=0.0):
    th, wth, s_or, w_or = model_nodes(mdl)
    rows = _rows(mdl)
    af_c, q = mdl.af_c[rows], mdl.q[mdl.iq]
    qz_vac = M.K0 * (np.sin(np.radians(mdl.ai)) + np.sin(np.radians(np.maximum(af_c, 0.0))))
    pRj, pHj, pwj = _size_nodes(pop)
    Rm = float(np.sum(pwj * pRj))
    if DWBA == 0:
        out = 0
        for th_, wt_ in zip(th, wth):
            ct, st = np.cos(th_), np.sin(th_)
            for s_, w_ in zip(s_or, w_or):
                qzl = q[None, :] * st * s_ + qz_vac[:, None] * ct
                qpl = np.sqrt(np.maximum(q[None, :] ** 2 + qz_vac[:, None] ** 2 - qzl ** 2, 0.0))
                acc = 0
                for j0 in range(0, len(pRj), 32):
                    Rj, Hj, wj = pRj[j0:j0 + 32], pHj[j0:j0 + 32], pwj[j0:j0 + 32]
                    F = M.ff_arr_nodes(qpl, qzl, Rj, Hj)
                    I = F.real ** 2 + F.imag ** 2
                    I = I * _lma(mdl, qpl, Rj, Rm, D, omega, lma, eta, ct) if lma else I
                    acc = acc + np.einsum("k,kaq->aq", wj, I)
                out = out + wt_ * w_ * acc
        return n * np.exp(_interp_to_full(mdl, af_c, np.log(np.maximum(out, 1e-300))))
    amp, bj, nb, Q12, nt = _dwba_channels(mdl, af_c)
    G = np.zeros((nb, nt, nt, len(af_c), len(q)), complex)
    s_list = list(zip(s_or, w_or))
    smap = {}
    for s_, w_ in s_list:
        k_ = round(float(s_), 12)
        smap[k_] = smap.get(k_, 0.0) + w_
    for th_, wt_ in zip(th, wth):
        ct, st = np.cos(th_), np.sin(th_)
        done = set()
        for s_, w_ in s_list:
            key = round(abs(float(s_)), 12)
            if key in done:
                continue
            done.add(key)

            signs = [1.0] if key == 0.0 else [sg for sg in (1.0, -1.0) if round(sg * key, 12) in smap]
            ch = max(1, int(2.5e6 // (nb * Q12.shape[1] * len(af_c) * len(q) * 20)))
            for j0 in range(0, len(pRj), ch):
                Rj, Hj, wj = pRj[j0:j0 + ch], pHj[j0:j0 + ch], pwj[j0:j0 + ch]
                Fp = _f_pair(Q12, q, st, ct, key, Rj, Hj, nt)
                for sg in signs:
                    w_or_ = wt_ * smap[round(sg * key, 12)]
                    qzl = sg * key * q[None, :] * st + qz_vac[:, None] * ct
                    qpl = np.sqrt(np.maximum(q[None, :] ** 2 + qz_vac[:, None] ** 2 - qzl ** 2, 0.0))
                    Sx = _lma(mdl, qpl, Rj, Rm, D, omega, lma, eta, ct)
                    wS = (wj[:, None, None] * Sx)[:, None, None]
                    F = Fp[sg]
                    G += w_or_ * np.einsum("jbtaq,jbuaq->btuaq", F * wS, F.conj())
    return _contract(mdl, G, amp, bj, nb, nt, af_c, q, n)


def cap_quadrature(R, H, qmax, Qmax, scale=1.0):
    'Level-5 nodes of a hemispheroid (base radius R, height H; (rho/R)^2 + (h/H)^2 <= 1, 0 <= h <= H) in its cap'
    sph = FFTEST == "sphere"
    Hh = 2.0 * R if sph else H
    kap = qmax + Qmax
    n_h = int(np.ceil(scale * max(32.0, 0.8 * kap * Hh + 16)))
    n_b = int(np.ceil(scale * (0.8 * kap * R + 16)))
    xg, wg = np.polynomial.legendre.leggauss(n_h)
    if sph:
        gam, dg = 0.5 * np.pi * xg, 0.5 * np.pi * wg
        h, hs = R * np.sin(gam), R
    else:
        gam, dg = 0.25 * np.pi * (xg + 1), 0.25 * np.pi * wg
        h, hs = H * np.sin(gam), H
    xb, wb = np.polynomial.legendre.leggauss(n_b)
    bet, db = 0.5 * np.pi * xb, 0.5 * np.pi * wb
    r = R * np.cos(gam)
    u = (r[:, None] * np.sin(bet)[None, :]).ravel()
    V = (r[:, None] * np.cos(bet)[None, :]).ravel()
    w = np.broadcast_to((h if sph else h - 0.375 * H)[:, None], (n_h, n_b)).ravel()
    W = ((hs * np.cos(gam) * dg)[:, None] * (r[:, None] * np.cos(bet)[None, :] * db[None, :])).ravel()
    return u, V, w, W


def cap_quadrature_3d(R, H, qmax, Qmax, scale=1.0):
    'Independent pointwise volume nodes of the same hemispheroid (volume_check, N8e reference): u, v in the base'
    n_h = int(np.ceil(scale * (0.5 * Qmax * H + 12)))
    n_r = int(np.ceil(scale * (0.5 * qmax * R + 10)))
    n_p = 2 * int(np.ceil(scale * (qmax * R + 8)))
    xh, wh = np.polynomial.legendre.leggauss(n_h)
    xh, wh = 0.5 * (xh + 1), 0.5 * wh
    xr, wr = np.polynomial.legendre.leggauss(n_r)
    xr, wr = 0.5 * (xr + 1), 0.5 * wr
    psi = 2 * np.pi * (np.arange(n_p) + 0.5) / n_p
    rmax = R * np.sqrt(np.maximum(1 - xh ** 2, 0.0))
    rho = rmax[:, None] * np.sqrt(xr)[None, :]
    Whr = (H * wh * 0.5 * rmax ** 2)[:, None] * wr[None, :]
    u = (rho[..., None] * np.cos(psi)).ravel()
    v = (rho[..., None] * np.sin(psi)).ravel()
    w = np.broadcast_to((H * xh - 0.375 * H)[:, None, None], rho.shape + (n_p,)).ravel()
    W = np.broadcast_to(Whr[..., None] * (2 * np.pi / n_p), rho.shape + (n_p,)).ravel()
    return u, v, w, W


def _vol3d_P(quad, q, ct, st, s_):
    'Pointwise phase factors W_p exp(-i q y_p) (n_p, nq) of cap_quadrature_3d for one orientation.'
    u, v, w, W = quad
    y = (w * st + u * ct) * s_ + v * np.sqrt(max(1 - s_ * s_, 0.0))
    return W[:, None] * np.exp(-1j * y[:, None] * q[None, :])


def _level5_setup(mdl, rows):
    amp = mdl._amp
    t, wt = amp["t"], amp["wt"]
    keep = np.nonzero(wt > 1e-14 * wt.max())[0]
    zmid = np.r_[0.0, np.cumsum(t)][:-1] + 0.5 * t
    Qmax = 2 * float(np.max(np.abs(amp["k"][:, np.r_[0, 1 + rows]])))
    return amp, keep, zmid[keep], wt[keep], Qmax


def _level5_E(amp, zc, quad, ct, st, cols):
    'Fields E_i E_f (nc, n_p, n_a) at every node for the centroid depths zc (depends on theta only; the depth of a'
    u, _, w, W = quad
    z_up = w * ct - u * st
    E, _ = _fields_at(amp, (zc[:, None] - z_up[None, :]).ravel(), cols)
    return E.reshape(len(zc), len(u), -1)


def _level5_P(quad, q, ct, st, s_):
    'Weighted factors (n_p, nq) for one orientation (theta, s = sin phi): W 2V sinc(q c V) exp(-i q (w st + u ct) s),'
    u, V, w, W = quad
    c = np.sqrt(max(1 - s_ * s_, 0.0))
    x = V[:, None] * (q * c)[None, :]
    return (2 * W * V)[:, None] * np.sinc(x / np.pi) * np.exp(-1j * ((w * st + u * ct) * s_)[:, None] * q[None, :])


def _level5_out(mdl, af_c, rows, T, n):
    'Level-5 island term from the summed T on (rows, coarse q): n exp(interp log(T / <|E_i E_f|^2>)) on the full grid.'
    EiEf = np.asarray(mdl._EiEf, float)[rows]
    ratio = T / np.maximum(EiEf[:, None], 1e-300)
    return n * np.exp(_interp_to_full(mdl, af_c, np.log(np.maximum(ratio, 1e-300))))


def _born_level5(mdl, pop, n, D, omega, lma, eta, nested, _contrib=None):
    'Level 5: exact first-order DWBA in the stratified reference, the field at every node from the slice (or vacuum)'
    rows = _rows(mdl)
    af_c, q = mdl.af_c[rows], mdl.q[mdl.iq]
    qz_vac = M.K0 * (np.sin(np.radians(mdl.ai)) + np.sin(np.radians(np.maximum(af_c, 0.0))))
    pRj, pHj, pwj = _size_nodes(pop)
    Rm = float(np.sum(pwj * pRj))
    amp, keep, zc, wc, Qmax = _level5_setup(mdl, rows)
    cols = 1 + rows
    na, nq = len(af_c), len(q)
    T = np.zeros((na, nq))

    def S_at(Rj, ct, st, s_):
        qzl = s_ * q[None, :] * st + qz_vac[:, None] * ct
        qpl = np.sqrt(np.maximum(q[None, :] ** 2 + qz_vac[:, None] ** 2 - qzl ** 2, 0.0))
        return _lma(mdl, qpl, np.atleast_1d(Rj), Rm, D, omega, lma, eta, ct)[0]
    if _contrib is not None and not nested:
        th, wth, s_or, w_or = model_nodes(mdl)
        Tn = _contrib["T_nodes"] = np.zeros((len(th), len(s_or), na, nq))
    for Rj, Hj, wj in zip(pRj, pHj, pwj):
        quad = cap_quadrature(Rj, Hj, float(q.max()), Qmax, L5_SCALE)
        chunk = max(1, int(2e6 // (len(quad[0]) * (1 + len(cols)))))
        if not nested:
            th, wth, s_or, w_or = model_nodes(mdl)
            for it, (th_, wt_) in enumerate(zip(th, wth)):
                ct, st = np.cos(th_), np.sin(th_)
                Ps = [_level5_P(quad, q, ct, st, s_) for s_ in s_or]
                Ss = [wt_ * w_ * wj * S_at(Rj, ct, st, s_) for s_, w_ in zip(s_or, w_or)]
                for c0 in range(0, len(zc), chunk):
                    Et = _level5_E(amp, zc[c0:c0 + chunk], quad, ct, st, cols).transpose(0, 2, 1)
                    w_c = wc[c0:c0 + chunk]
                    for i_s, (P, Sw) in enumerate(zip(Ps, Ss)):
                        A = np.matmul(Et, P)
                        X = Sw * np.einsum("c,caq->aq", w_c, A.real ** 2 + A.imag ** 2)
                        T += X
                        if _contrib is not None:
                            Tn[it, i_s] += X
            continue
        thc, sc, thf, wthf, sf, wsf, Lt, Ls = nested_nodes(mdl)
        Sf = [[wt_ * w_ * wj * S_at(Rj, np.cos(th_), np.sin(th_), float(s_)) for s_, w_ in zip(sf, wsf)]
              for th_, wt_ in zip(thf, wthf)]
        Pc = [[_level5_P(quad, q, np.cos(th_), np.sin(th_), float(s_)) for s_ in sc] for th_ in thc]
        if _contrib is not None:
            ck = np.zeros((len(thf), len(sf)))
        for c0 in range(0, len(zc), chunk):
            zcc, w_c = zc[c0:c0 + chunk], wc[c0:c0 + chunk]
            Ac = np.empty((len(thc), len(sc), len(zcc), na, nq), complex)
            for i, th_ in enumerate(thc):
                Et = _level5_E(amp, zcc, quad, np.cos(th_), np.sin(th_), cols).transpose(0, 2, 1)
                for jj in range(len(sc)):
                    Ac[i, jj] = np.matmul(Et, Pc[i][jj])
            for k in range(len(thf)):
                A_k = np.tensordot(Lt[k], Ac, axes=(0, 0))
                Am = np.tensordot(Ls, A_k, axes=(1, 0))
                Im = np.einsum("c,mcaq->maq", w_c, Am.real ** 2 + Am.imag ** 2)
                T += sum(Sf[k][m_] * Im[m_] for m_ in range(len(sf)))
                if _contrib is not None:
                    ck[k] += n * np.einsum("maq,maq->m", np.asarray(Sf[k]) * _contrib["Wcell"], Im)
        if _contrib is not None:
            for k in range(len(thf)):
                for m_ in range(len(sf)):
                    _contrib["nodes"].append((float(Rj), float(Hj), float(thf[k]), float(sf[m_]),
                                              float(wthf[k] * wsf[m_] * wj), float(ck[k, m_])))
    if _contrib is not None and nested:
        _contrib["total"] = n * float(np.sum(_contrib["Wcell"] * T))
    return _level5_out(mdl, af_c, rows, T, n)


def _single_nodes(th, s_):
    'model_nodes replacement: one orientation (theta, s) with weight 1.'
    return lambda mdl: (np.array([th]), np.array([1.0]), np.array([s_]), np.array([1.0]))


def grid_model(mdl, af, qy):
    'A shallow copy of a model whose slice amplitudes (same metal-loaded layers and slice weights) are rebuilt on the'
    import copy
    amp0 = mdl._amp
    m2 = copy.copy(mdl)
    af = np.asarray(af, float)
    m2.af = af
    m2.af_c = af
    m2.a_all = np.r_[mdl.ai, af]
    m2.q = np.asarray(qy, float)
    m2.iq = np.arange(len(m2.q))
    m2.ia = np.arange(len(af))
    m2._born_cache, m2._tnodes, m2._nnodes = {}, None, None
    layers = [(d, t) for d, t in zip(amp0["d"], amp0["t"])]
    kzs, A, B, ts = M.stack(m2.a_all, layers, mdl.d_sub)
    N = len(layers)
    k = np.array(kzs[1:N + 1])
    t = np.array(ts[1:N + 1])[:, None]
    m2._amp = dict(k=k, Cd=np.array(A[1:N + 1]) * np.exp(1j * k * 0.5 * t), Cu=np.array(B[1:N + 1]) * np.exp(-1j * k * 0.5 * t),
                   d=np.array(amp0["d"]), wt=amp0["wt"], t=t[:, 0], wm=amp0["wm"],
                   vac=(np.asarray(kzs[0]), np.asarray(A[0]), np.asarray(B[0])))
    E_ = m2._amp["Cd"] + m2._amp["Cu"]
    m2._EiEf = np.sum(amp0["wt"][:, None] * np.abs(E_[:, :1]) ** 2 * np.abs(E_[:, 1:]) ** 2, axis=0)
    return m2


def _one_size(R, H):
    import types
    return types.SimpleNamespace(R=float(R), s=0.0, a=float(H) / float(R), Rj=np.array([float(R)]), Hj=np.array([float(H)]),
                                 wj=np.array([1.0]))


def level_single(m2, R, H, th, s_):
    'Island term of the level in force for one size and one orientation, S = 1, n = 1, on the full grid of m2:'
    global model_nodes, ROWS
    q0, mn, r0 = M.QUAD, model_nodes, ROWS
    M.QUAD, model_nodes, ROWS = "gh", _single_nodes(th, s_), "full"
    try:
        pop = _one_size(R, H)
        m2._tnodes = None
        if DWBA == 5:
            return np.asarray(_born_level5(m2, pop, 1.0, 1.0, 0.0, False, 0.0, nested=False))
        if DWBA in (1, 4):
            return np.asarray(_born_plain(m2, pop, 1.0, 1.0, 0.0, False))
        raise SystemExit("level_single: DWBA level 1, 4 or 5 only")
    finally:
        M.QUAD, model_nodes, ROWS = q0, mn, r0


def level_group(m2, R, H, th, ss):
    'level_single(m2, R, H, th, s) for every s of ss (one size, one theta). Level 5: one _born_level5 call on the'
    global model_nodes, ROWS
    if DWBA in (1, 4):
        return [level_single(m2, R, H, th, s_) for s_ in ss]
    if DWBA != 5:
        raise SystemExit("level_group: DWBA level 1, 4 or 5 only")
    ss = np.asarray(ss, float)
    q0, mn, r0 = M.QUAD, model_nodes, ROWS
    M.QUAD, ROWS = "gh", "full"
    model_nodes = lambda mdl: (np.array([th]), np.array([1.0]), ss, np.ones(len(ss)))
    try:
        m2._tnodes = None
        acc = {}
        _born_level5(m2, _one_size(R, H), 1.0, 1.0, 0.0, False, 0.0, nested=False, _contrib=acc)
        rows = np.arange(len(m2.af))
        return [np.asarray(_level5_out(m2, m2.af_c[rows], rows, acc["T_nodes"][0, i], 1.0)) for i in range(len(ss))]
    finally:
        M.QUAD, model_nodes, ROWS = q0, mn, r0


def _volume_group(m2, R, H, th, ss, scale=1.0):
    'volume_single for every s of ss (one size, one theta): the fields once per centroid chunk.'
    rows = np.arange(len(m2.af))
    q = m2.q
    amp, keep, zc, wc, Qmax = _level5_setup(m2, rows)
    cols = 1 + rows
    quad = cap_quadrature_3d(R, H, float(q.max()), Qmax, scale)
    ct, st = np.cos(th), np.sin(th)
    Ps = [_vol3d_P(quad, q, ct, st, float(s_)) for s_ in ss]
    T = [np.zeros((len(rows), len(q))) for _ in Ps]
    ch = max(1, int(2e6 // (len(quad[0]) * (1 + len(cols)))))
    for c0 in range(0, len(zc), ch):
        Et = _level5_E(amp, zc[c0:c0 + ch], quad, ct, st, cols).transpose(0, 2, 1)
        for i, P in enumerate(Ps):
            A = np.matmul(Et, P)
            T[i] += np.einsum("c,caq->aq", wc[c0:c0 + ch], A.real ** 2 + A.imag ** 2)
    EiEf = np.maximum(np.asarray(m2._EiEf, float)[:, None], 1e-300)
    return [Ti / EiEf for Ti in T]


def volume_single(m2, R, H, th, s_, scale=1.0):
    'Independent pointwise volume integral (cap_quadrature_3d: every node takes the field of the slice or vacuum that'
    return _volume_group(m2, R, H, th, [s_], scale)[0]


def volume_check(mdl, nodes, qy, af, selfcheck=None):
    'A2d (ii) hook: single-orientation intensities (n_nodes, n_qy, n_af) of the level in force (I_level) and of the'
    m2 = grid_model(mdl, af, qy)
    n = len(nodes["R"])
    groups = {}
    for j in range(n):
        groups.setdefault((float(nodes["R"][j]), float(nodes["H"][j]), float(nodes["theta"][j])), []).append(j)
    sel = [int(j) for j in (selfcheck if selfcheck is not None else [])]
    Il, Iv, Iv2 = [None] * n, [None] * n, {}
    for (R, H, th), js in groups.items():
        ss = [float(nodes["s"][j]) for j in js]
        for j, a, b in zip(js, level_group(m2, R, H, th, ss), _volume_group(m2, R, H, th, ss, 1.0)):
            Il[j], Iv[j] = a.T, b.T
        js2 = [j for j in js if j in sel]
        if js2:
            for j, b in zip(js2, _volume_group(m2, R, H, th, [float(nodes["s"][j]) for j in js2], 2.0)):
                Iv2[j] = b.T
    out = dict(I_level=np.array(Il), I_vol=np.array(Iv))
    if selfcheck is not None:
        out["I_vol2"] = np.array([Iv2[j] for j in sel]).reshape((len(sel), len(qy), len(af)))
    return out


def _interp_matrix(x_to, x_from):
    'L with np.interp(x_to, x_from, y) == L @ y for every y (linear, end values held; x_from ascending).'
    E = np.eye(len(x_from))
    return np.stack([np.interp(x_to, x_from, E[c]) for c in range(len(x_from))], axis=1)


def node_contributions(mdl, m, ok):
    'A2d (ii) hook (operational_definitions (5)): one entry per size x fine-orientation node of the island term (S and'
    if ORIENT_MODE != "nested" or DWBA not in (1, 4, 5) or QZVAC or FFTEST:
        raise SystemExit("node_contributions: nested orientation mode, DWBA level 1, 4 or 5, no test hooks")
    rows = _rows(mdl)
    af_c, q = mdl.af_c[rows], mdl.q[mdl.iq]
    okc = np.asarray(ok)
    cnt = okc.reshape(-1, *okc.shape[-2:]).sum(axis=0).astype(float)
    if cnt.shape != (len(mdl.af), len(mdl.q)):
        raise ValueError("node_contributions: ok must end in (n_af, n_q) of the model grid")
    Lr = _interp_matrix(mdl.af_c, af_c)
    CQ = cnt @ _interp_matrix(np.log(mdl.q), np.log(q))
    if DWBA == 5:
        EiEf = np.asarray(mdl._EiEf, float)
        cw = dict(Wcell=(Lr.T @ (EiEf[:, None] * CQ)) / np.maximum(EiEf[rows][:, None], 1e-300))
    else:
        amp, bj, nb, _, nt = _dwba_channels(mdl, af_c)
        C, _ = channels(amp, 1 + np.arange(len(mdl.af)))
        if nt == 1:
            C = C.sum(axis=0, keepdims=True)
        Weff = np.empty((nb, nt, nt, len(af_c), len(q)), complex)
        for b in range(nb):
            sel = bj == b
            Wb = np.einsum("s,tsf,usf->tuf", amp["wt"][sel], C[:, sel], C[:, sel].conj())
            Weff[b] = np.einsum("fa,tuf,fq->tuaq", Lr, Wb, CQ)
        cw = dict(Weff=Weff)
    pops = [("S", m.S, m.nS, True, m.D, m.omega, m.eta_hd)]
    if m.fL > 0:
        pops.append(("L", m.L, m.nL, False, 1.0, 0.0, 0.0))
    out = dict(pop=[], R=[], H=[], theta=[], s=[], w=[], contrib=[])
    check = {}
    for nm, pop, n, lma, D, omega, eta in pops:
        acc = dict(cw, nodes=[])
        if DWBA == 5:
            _born_level5(mdl, pop, n, D, omega, lma, eta, nested=True, _contrib=acc)
        else:
            _born_nested(mdl, pop, n, D, omega, lma, eta, _contrib=acc)
        tot, sn = acc["total"], float(sum(v[5] for v in acc["nodes"]))
        rel = abs(sn - tot) / max(abs(tot), 1e-300)
        check[nm] = dict(sum_nodes=sn, total=tot, rel=rel, n_nodes=len(acc["nodes"]))
        if not rel <= 1e-9:
            raise AssertionError(f"node_contributions: {nm} node shares {sn} != term {tot} (rel {rel:.1e})")
        for R, H, th_, s_, w_, c_ in acc["nodes"]:
            for k_, v_ in zip(("pop", "R", "H", "theta", "s", "w", "contrib"), (nm, R, H, th_, s_, w_, c_)):
                out[k_].append(v_)
    mdl._a2d_contrib_check = check
    return {k: np.asarray(v) for k, v in out.items()}


def lobatto(n, a, b):
    'n + 1 Chebyshev-Lobatto nodes on [a, b] (ascending, symmetric about the centre).'
    return a + (b - a) * 0.5 * (1 - np.cos(np.pi * np.arange(n + 1) / n))


def bary_matrix(xc, xf):
    'Barycentric interpolation matrix (len(xf), len(xc)) from Chebyshev-Lobatto nodes xc to points xf.'
    n = len(xc) - 1
    wb = np.array([(-1) ** j * (0.5 if j in (0, n) else 1.0) for j in range(n + 1)])
    L = np.zeros((len(xf), len(xc)))
    for i, x in enumerate(xf):
        d = x - xc
        hit = np.abs(d) < 1e-14
        if hit.any():
            L[i, np.argmax(hit)] = 1.0
        else:
            t = wb / d
            L[i] = t / t.sum()
    return L


def nested_nodes(mdl):
    'Coarse (F) and fine (S, weights) orientation nodes of the two-level quadrature (cached).'
    key = _settings() + (mdl.theta_max,)
    if getattr(mdl, "_nnodes", None) is None or mdl._nnodes[0] != key:
        nFt, nFs = ORIENT
        nSt, nSp = ORIENT_S
        thc = lobatto(nFt, 0.0, np.radians(mdl.theta_max))
        sc = lobatto(nFs, -1.0, 1.0)
        sc[np.abs(sc) < 1e-15] = 0.0
        thf, wthf, sf, wsf = orientation_nodes(nSt, nSp, mdl.theta_max, az="full", phi="uni")
        mdl._nnodes = (key, (thc, sc, thf, wthf, sf, wsf, bary_matrix(thc, thf), bary_matrix(sc, sf)))
    return mdl._nnodes[1]


def _f_pair(Q12, q, st, ct, sabs, Rj, Hj, nt):
    'Channel amplitudes F (nj, nb, nt, na, nq) at s = +sabs and s = -sabs; channels 3, 4 reuse the Bessel parts of'
    if FFTEST == "sphere":
        return _f_pair_sphere(Q12, q, Rj, nt)
    Kn, g = {}, {}
    for sg in (1.0, -1.0):
        qy_n = sg * sabs * q[None, :] * st
        Kn[sg] = -(qy_n[None, None] + Q12[..., None] * ct)
        Kt2 = q[None, None, None, :] ** 2 + Q12[..., None] ** 2 - Kn[sg] ** 2
        g[sg] = _g_part(Kt2, Rj, Hj, None)
    out = {}
    for sg in (1.0, -1.0):
        F12 = _ff_from_g(g[sg], Kn[sg], Rj, Hj)
        if nt == 4:
            F3 = _ff_from_g(g[-sg][:, :, 1], -Kn[-sg][:, 1], Rj, Hj)
            F4 = _ff_from_g(g[-sg][:, :, 0], -Kn[-sg][:, 0], Rj, Hj)
            out[sg] = np.concatenate([F12, F3[:, :, None], F4[:, :, None]], axis=2)
        else:
            out[sg] = F12
    return out


def _f_pair_sphere(Q12, q, Rj, nt):
    'Test hook IAM_TIO2_FFTEST=sphere: the channel amplitudes of _f_pair for a sphere of radius Rj about its centre,'
    Rj = np.asarray(Rj, float).reshape(-1, 1, 1, 1, 1)
    x = np.sqrt(q[None, None, None, :] ** 2 + Q12[..., None] ** 2 + 0j)[None] * Rj
    small = np.abs(x) < 1e-2
    xs = np.where(small, 1.0, x)
    f = np.where(small, 1 - x * x / 10 + x ** 4 / 280, 3 * (np.sin(xs) - xs * np.cos(xs)) / xs ** 3)
    F = (4.0 / 3.0 * np.pi * Rj ** 3) * f
    if nt == 4:
        F = np.concatenate([F, F[:, :, 1:2], F[:, :, 0:1]], axis=2)
    return {1.0: F, -1.0: F}


def _born_nested(mdl, pop, n, D, omega, lma, eta, _ref_loop=False, _contrib=None):
    'Two-level orientation quadrature: F on the coarse Chebyshev-Lobatto (theta, s) nodes, separable barycentric'
    thc, sc, thf, wthf, sf, wsf, Lt, Ls = nested_nodes(mdl)
    rows = _rows(mdl)
    af_c, q = mdl.af_c[rows], mdl.q[mdl.iq]
    qz_vac = M.K0 * (np.sin(np.radians(mdl.ai)) + np.sin(np.radians(np.maximum(af_c, 0.0))))
    pRj, pHj, pwj = _size_nodes(pop)
    Rm = float(np.sum(pwj * pRj))
    if DWBA == 0:
        nb = nt = 1
        Q12 = None
    else:
        amp, bj, nb, Q12, nt = _dwba_channels(mdl, af_c)
    if _contrib is not None and (DWBA == 0 or _ref_loop):
        raise SystemExit("_born_nested: node shares for levels 1, 4 only")
    na, nq = len(af_c), len(q)
    G = np.zeros((nb, nt, nt, na, nq), complex) if DWBA else np.zeros((na, nq))
    per = max(1, nb * max(nt, 2) * na * nq * 20)
    ch = max(1, int(2.5e6 // per))
    for j0 in range(0, len(pRj), ch):
        Rj, Hj, wj = pRj[j0:j0 + ch], pHj[j0:j0 + ch], pwj[j0:j0 + ch]
        nj = len(Rj)
        shp = (nj, na, nq) if DWBA == 0 else (nj, nb, nt, na, nq)
        Fc = np.empty((len(thc), len(sc)) + shp, complex)
        for i, th_ in enumerate(thc):
            ct, st = np.cos(th_), np.sin(th_)
            if DWBA == 0:
                for jj, s_ in enumerate(sc):
                    if i == 0 and jj > 0:
                        Fc[i, jj] = Fc[i, 0]
                        continue
                    qzl = q[None, :] * st * s_ + qz_vac[:, None] * ct
                    qpl = np.sqrt(np.maximum(q[None, :] ** 2 + qz_vac[:, None] ** 2 - qzl ** 2, 0.0))
                    Fc[i, jj] = M.ff_arr_nodes(qpl, qzl, Rj, Hj)
                continue
            if i == 0:
                Fp = _f_pair(Q12, q, st, ct, 0.0, Rj, Hj, nt)[1.0]
                Fc[i] = Fp[None]
                continue
            for jj, s_ in enumerate(sc):
                if s_ < 0:
                    continue
                Fp = _f_pair(Q12, q, st, ct, float(s_), Rj, Hj, nt)
                Fc[i, jj] = Fp[1.0]
                jm = int(np.argmin(np.abs(sc + s_)))
                Fc[i, jm] = Fp[-1.0]
        nsf = len(sf)
        for k, (th_, wt_) in enumerate(zip(thf, wthf)):
            ct, st = np.cos(th_), np.sin(th_)
            Fk = np.tensordot(Lt[k], Fc, axes=(0, 0))
            Fm = np.tensordot(Ls, Fk, axes=(1, 0))
            qzl = q[None, None, :] * st * sf[:, None, None] + qz_vac[None, :, None] * ct
            qpl = np.sqrt(np.maximum(q[None, None, :] ** 2 + qz_vac[None, :, None] ** 2 - qzl ** 2, 0.0))
            if lma and mdl.interf == "hd":
                Sx = np.array([_lma(mdl, qpl[m_], Rj, Rm, D, omega, lma, eta, ct) for m_ in range(nsf)])
            elif lma:
                Sx = M.s_para(qpl[:, None], M.lma_D(D / np.sqrt(max(ct, 1e-3)), Rj, Rm)[None, :, None, None], omega)
            else:
                Sx = np.ones((nsf, nj) + qpl.shape[1:])
            Wt = (wt_ * wsf)[:, None, None, None] * wj[None, :, None, None] * Sx
            if _ref_loop:
                for m_ in range(nsf):
                    F = Fm[m_]
                    if DWBA == 0:
                        G += np.einsum("j,jaq->aq", np.ones(nj), (F.real ** 2 + F.imag ** 2) * Wt[m_])
                    else:
                        G += np.einsum("jbtaq,jbuaq->btuaq", F * Wt[m_][:, None, None], F.conj())
                continue
            if DWBA == 0:
                G += np.einsum("mjaq,mjaq->aq", Wt, Fm.real ** 2 + Fm.imag ** 2)
            else:

                A = (Fm * Wt[:, :, None, None]).transpose(2, 4, 5, 3, 0, 1).reshape(nb * na * nq, nt, nsf * nj)
                B = Fm.conj().transpose(2, 4, 5, 0, 1, 3).reshape(nb * na * nq, nsf * nj, nt)
                G += np.matmul(A, B).reshape(nb, na, nq, nt, nt).transpose(0, 3, 4, 1, 2)
                if _contrib is not None:
                    Y = np.einsum("btuaq,mjbtaq->mjbuaq", _contrib["Weff"], Fm)
                    Z = np.einsum("mjbuaq,mjbuaq->mjaq", Y, Fm.conj()).real
                    cm = n * np.einsum("mjaq,mjaq->mj", Wt, Z)
                    for j_ in range(nj):
                        for m_ in range(nsf):
                            _contrib["nodes"].append((float(Rj[j_]), float(Hj[j_]), float(th_), float(sf[m_]),
                                                      float(wt_ * wsf[m_] * wj[j_]), float(cm[m_, j_])))
    if DWBA == 0:
        return n * np.exp(_interp_to_full(mdl, af_c, np.log(np.maximum(G, 1e-300))))
    if _contrib is not None:
        _contrib["total"] = n * float(np.einsum("btuaq,btuaq->", _contrib["Weff"], G).real)
    return _contract(mdl, G, amp, bj, nb, nt, af_c, q, n)


def _contract(mdl, G, amp, bj, nb, nt, af_c, q, n):
    'G^b_tu (coarse rows) -> full alpha_f grid, contracted with the per-slice channel weights; / <|E_i E_f|^2>.'
    if ROWS != "full":
        Gf = np.empty((nb, nt, nt, len(mdl.af), len(q)), complex)
        for b in range(nb):
            for t_ in range(nt):
                for u in range(nt):
                    for jq in range(len(q)):
                        g_ = G[b, t_, u, :, jq]
                        Gf[b, t_, u, :, jq] = np.interp(mdl.af_c, af_c, g_.real) + 1j * np.interp(mdl.af_c, af_c, g_.imag)
    else:
        Gf = G
    C, _ = channels(amp, 1 + np.arange(len(mdl.af)))
    if nt == 1:
        C = C.sum(axis=0, keepdims=True)
    wtv = amp["wt"]
    T = np.zeros((len(mdl.af), len(q)))
    for b in range(nb):
        sel = bj == b
        Cb = C[:, sel]
        W = np.einsum("s,tsa,usa->tua", wtv[sel], Cb, Cb.conj())
        T += np.einsum("tua,tuaq->aq", W, Gf[b]).real
    EiEf = np.asarray(mdl._EiEf, float)
    ratio = T / np.maximum(EiEf[:, None], 1e-300)
    lo = np.log(np.maximum(ratio, 1e-300))
    A2 = np.array([np.interp(np.log(mdl.q), np.log(q), lo[i]) for i in range(lo.shape[0])])
    return n * np.exp(A2)


def network_vertical(mdl, sz):
    'Four-term in-slice vertical factor of the network term on the full alpha_f grid:'
    amp = mdl._amp
    full = 1 + np.arange(len(mdl.af))
    C, sgn = channels(amp, full)
    ki, kf = amp["k"][:, :1], amp["k"][:, full]
    Qz = np.array([s0 * ki + s1 * kf for s0, s1 in sgn])
    if QZVAC:
        qv = M.K0 * (np.sin(np.radians(mdl.ai)) + np.sin(np.radians(np.maximum(mdl.af_c, 0.0))))
        Qz = np.broadcast_to(qv[None, None, :], Qz.shape) + 0j
    wtv = amp["wt"]
    V = np.zeros(len(mdl.af))
    for t_ in range(4):
        for u in range(4):
            Kc = 0.5 * (Qz[t_] + Qz[u].conj())
            V += np.sum(wtv[:, None] * (C[t_] * C[u].conj() * np.exp(-(sz * Kc) ** 2)), axis=0).real
    return V


def _fields_at(amp, zs, cols):
    'Total field product E_i(z) E_f(z) at depths zs (nm below the stack top) from the per-slice Parratt amplitudes:'
    t = amp["t"]
    ztop = np.r_[0.0, np.cumsum(t)]
    zmid = ztop[:-1] + 0.5 * t
    s = np.clip(np.searchsorted(ztop, zs, side="right") - 1, 0, len(t) - 1)
    dz = (zs - zmid[s])[:, None]
    k = amp["k"]

    def E(c):
        ix = (s[:, None], np.asarray(c)[None, :])
        kk = k[ix]
        return amp["Cd"][ix] * np.exp(1j * kk * dz) + amp["Cu"][ix] * np.exp(-1j * kk * dz)
    out = E([0]) * E(cols)
    above = zs < 0
    if np.any(above) and "vac" in amp:
        k0, A0, B0 = amp["vac"]
        z = zs[above][:, None]

        def Ev(c):
            return A0[c][None] * np.exp(1j * k0[c][None] * z) + B0[c][None] * np.exp(-1j * k0[c][None] * z)
        out[above] = Ev([0]) * Ev(cols)
        s = s.copy()
        s[above] = -1
    return out, s


NET_NSUB, NET_CUT, NET_PAD = 4, 1e-10, 4.0


def network_double_integral(mdl, sz, nsub=None, cut=None, pad_sig=None, smooth=True):
    'IAM_TIO2_DWBA_NET=2 (amendment_1): the network vertical factor as the direct depth double integral of the'
    nsub = NET_NSUB if nsub is None else nsub
    cut = NET_CUT if cut is None else cut
    pad_sig = NET_PAD if pad_sig is None else pad_sig
    amp = mdl._amp
    t, w = amp["t"], amp["wt"]
    wm = amp.get("wm", w)
    if not np.any(w > 0):
        return np.zeros(len(mdl.af))
    keep = np.nonzero(w > cut * w.max())[0]
    lo_s, hi_s = keep.min(), keep.max()
    pad = int(np.ceil(pad_sig * max(sz, 1e-6) / max(t[lo_s:hi_s + 1].min(), 1e-6))) + 1
    lo_s, hi_s = max(lo_s - pad, 0), min(hi_s + pad, len(t) - 1)
    ztop = np.r_[0.0, np.cumsum(t)]
    xg, wg = np.polynomial.legendre.leggauss(nsub)
    zs = np.concatenate([ztop[s] + 0.5 * t[s] * (xg + 1) for s in range(lo_s, hi_s + 1)])
    dzs = np.concatenate([0.5 * t[s] * wg for s in range(lo_s, hi_s + 1)])
    cols = 1 + np.arange(len(mdl.af))
    a, _ = _fields_at(amp, zs, cols)
    rho_s = w / np.maximum(t, 1e-12)
    rn = rho_s[np.clip(np.searchsorted(ztop, zs, side="right") - 1, 0, len(t) - 1)]
    if smooth and mdl.tio2_bury is not None:
        z0, wb = mdl.tio2_bury
        g = np.exp(-0.5 * ((zs - z0) / max(wb, 1e-3)) ** 2)
        gs = np.exp(-0.5 * ((np.r_[0.0, np.cumsum(t)][:-1] + 0.5 * t - z0) / max(wb, 1e-3)) ** 2)
        sl = np.clip(np.searchsorted(ztop, zs, side="right") - 1, 0, len(t) - 1)
        cfac = np.where(wm > 0, w / np.where(wm > 0, wm, 1.0), 1.0)[sl]
        rn = g * (np.sum(wm) / max(np.sum(gs * t), 1e-300)) * cfac
    if sz <= 0:
        return np.sum((rn * dzs)[:, None] * np.abs(a) ** 2, axis=0)
    sw = np.sqrt(rn) * dzs
    Kmat = np.exp(-(zs[:, None] - zs[None, :]) ** 2 / (4 * sz * sz)) * (sw[:, None] * sw[None, :])
    Kmat /= 2 * sz * np.sqrt(np.pi)
    return np.sum(a * (Kmat @ a.conj()), axis=0).real


def contrast_weights(mdl, phm, fills, rho_T, rho_M):
    'Per-slice weight factor of the metal terms for IAM_TIO2_CONTRAST (fixed-parameter sensitivity only):'
    phi = np.asarray(phm, float) * mdl.nu_sub
    fills = np.asarray(fills, float)[: len(phi)]
    if CONTRAST == "em":
        return np.abs(rho_M - (fills * rho_T + phi * rho_M)) ** 2 / abs(rho_M) ** 2
    if CONTRAST == "pore":
        php = np.clip(phi / np.maximum(1 - fills, 1e-12), 0, 1)
        return (1 - php) ** 2
    return np.ones_like(phi)


def selftest():
    'Module-level checks (no data): jinc against scipy, orientation weights, channel pairing identity.'
    ok = True
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 16, 20000) + 1j * rng.uniform(-0.6, 0.6, 20000)
    X = z * z
    ref = np.where(np.abs(z) < 1e-4, 0.5, jv(1, z) / z)
    got = jinc_complex(X)
    err = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-3 * np.abs(ref).max())))
    ok &= err < 1e-6
    print("jinc_complex vs scipy jv: max rel err %.2e (<= 1e-6)" % err)
    xr = rng.uniform(0, 20, 1000)
    d = float(np.max(np.abs(jinc_X(xr ** 2 + 0j).real - M.jinc(xr))))
    ok &= d == 0.0
    print("real input path equals production jinc: max |diff| %.1e (== 0)" % d)
    for az, ph in (("half", "uni"), ("full", "gl"), ("full", "uni")):
        th, wth, s, ws = orientation_nodes(4, 5, 90.0, az=az, phi=ph)
        sym = az == "half" or np.allclose(np.sort(s), np.sort(-s))
        ok &= abs(ws.sum() - 1) < 1e-12 and abs(wth.sum() - 1) < 1e-12 and sym
        print("nodes", az, ph, "n_s", len(s), "sum w %.3f %.3f" % (wth.sum(), ws.sum()), "symmetric", sym)
    for g in ((32, 20), (64, 40), (8, 15)):
        _, _, s, ws = orientation_nodes(g[0], g[1], 90.0, az="full", phi="uni")
        sym = np.array_equal(s, -s[::-1]) and np.array_equal(ws, ws[::-1]) and abs(ws.sum() - 1) < 1e-12
        ok &= sym
        print("uni nodes", g, "n_s", len(s), "exactly symmetric", sym)
    rng2 = np.random.default_rng(2)
    zb = rng2.uniform(-250, 250, 20000) + 1j * rng2.uniform(-200, 200, 20000)
    zb = np.r_[zb, rng2.uniform(12, 18, 5000) + 1j * rng2.uniform(-3, 3, 5000)]
    refb = jv(1, zb) / zb
    errb = float(np.max(np.abs(jinc_complex(zb * zb) - refb) / np.abs(refb)))
    ok &= errb < 1e-6
    print("jinc_complex pointwise rel err, |Re z|<=250, |Im z|<=200 and the switch region: %.2e (<= 1e-6)" % errb)
    th, wth, s, ws = orientation_nodes(4, 5, 90.0, az="half")
    ph, wp = np.polynomial.legendre.leggauss(5)
    same = np.array_equal(s, np.sin(0.5 * np.pi * (ph + 1))) and np.array_equal(ws, 0.5 * wp)
    ok &= same
    print("half nodes equal production sin(ph), wph:", same)

    vol = []
    for R_, H_ in ((3.0, 2.1), (20.0, 5.0)):
        u_, V_, w_, W_ = cap_quadrature(R_, H_, 0.8, 1.7)
        u3, v3, w3, W3 = cap_quadrature_3d(R_, H_, 0.8, 1.7)
        vref = 2.0 / 3.0 * np.pi * R_ ** 2 * H_
        vol += [abs(np.sum(2 * W_ * V_) / vref - 1), abs(np.sum(W3) / vref - 1)]
        cen = abs(np.sum(2 * W_ * V_ * w_)) / vref / H_
        vol.append(cen)
    okv = max(vol) < 1e-12
    ok &= okv
    print("level-5 node volumes (chord-exact, 3-D) and centroid: max %.1e (< 1e-12)" % max(vol))
    qq = np.array([0.12, 0.3, 0.8])
    errf = 0.0
    for R_, H_ in ((3.0, 2.1), (6.0, 3.2)):
        qd = cap_quadrature(R_, H_, 0.8, 1.7, 2.0)
        q3 = cap_quadrature_3d(R_, H_, 0.8, 1.7, 3.0)
        for th_, s_ in ((0.0, 0.0), (0.7, 0.4), (1.3, -0.95)):
            ct, st = np.cos(th_), np.sin(th_)
            Fa = _level5_P(qd, qq, ct, st, s_).sum(axis=0)
            Fb = _vol3d_P(q3, qq, ct, st, s_).sum(axis=0)
            errf = max(errf, float(np.max(np.abs(Fa - Fb) / np.abs(Fb).max())))
    okf = errf < 1e-6
    ok &= okf
    print("level-5 chord-exact vs pointwise 3-D form factor (tilted caps, no fields): %.1e (< 1e-6)" % errf)
    print("tio2_dwba selftest", "ALL PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    selftest()
