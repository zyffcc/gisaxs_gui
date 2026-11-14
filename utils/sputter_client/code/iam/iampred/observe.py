'Readings taken from the counts of a window alone (no model, no network): direct observations that go into'

from __future__ import annotations

import numpy as np


KNEE_BAND = (0.20, 0.36)
KNEE_SHARE = 0.7
KNEE_EDGE = 0.9


def direct(s, w, eps=0.0, band=KNEE_BAND):
    'Lateral scale of a window read from the counts alone. Horizontal cut of the metal-induced rate (measured minus the'
    af = np.asarray(s.af, float) - float(eps)
    q = np.asarray(s.q, float)
    ok = w["mask"] & ((af >= band[0]) & (af < band[1]))[None, :, None]
    E = np.where(ok, w["E"], 0.0).sum((0, 1))
    Y = np.where(ok, w["Y"], 0.0).sum((0, 1))
    P = np.where(ok, w["cfac"] * s.I_pre * w["E"], 0.0).sum((0, 1))
    pos = E[E > 0]
    good = (E > 0.3 * (np.median(pos) if pos.size else 1.0)) & (Y - P > 3.0 * np.sqrt(Y + 1.0))
    idx = np.where(good)[0]
    if len(idx) < 8:
        return dict(q_knee=None, L_knee_nm=None, status="no signal", n_q=int(len(idx)))
    v = np.where(good, q * q * (Y - P) / np.maximum(E, 1e-12), np.nan)
    sm = np.full(len(q), np.nan)
    for i in idx:
        sm[i] = np.nanmean(v[max(i - 1, 0): i + 2])
    top = float(np.nanmax(sm[idx]))
    kg = int(idx[np.nanargmax(sm[idx])])
    if max(sm[idx[0]], sm[idx[1]]) >= KNEE_EDGE * top:
        k, st = kg, "at lower q limit"
    else:
        loc = [int(i) for n_, i in enumerate(idx) if 1 < n_ < len(idx) - 1 and sm[i] >= KNEE_SHARE * top
               and sm[i] >= sm[idx[n_ - 1]] and sm[i] >= sm[idx[n_ + 1]]]
        k = loc[0] if loc else kg
        st = "at upper q limit" if k >= idx[-1] - 1 else "knee"
    qk = float(q[k])
    if st == "knee" and np.isfinite(sm[k - 1]) and np.isfinite(sm[k + 1]):
        den = sm[k - 1] - 2 * sm[k] + sm[k + 1]
        if den < 0:
            qk = float(q[k] + 0.5 * (sm[k - 1] - sm[k + 1]) / den * (q[k + 1] - q[k]))
    return dict(q_knee=qk, L_knee_nm=float(2 * np.pi / qk), status=st, n_q=int(len(idx)))


FRINGE_AF = (0.36, 1.32)
FRINGE_Q = (0.06, 0.20)
GRAIN_Q = (0.35, 1.00)
FRINGE_D = (0.9, 2.2)
FRINGE_A = (0.25, 0.4, 0.55, 0.7, 0.85)
FRINGE_SCORE = 0.5
FRINGE_CELLS = 40


def fringe(s, w, h_nom, sin2_ac, k0, eps=0.0, q_band=FRINGE_Q):
    'Film thickness of a window read from the counts alone: the low-q diffuse scattering of a coalescing / closed'
    none = dict(d_nm=None, contrast=None, score=None, n_af=0)
    if str(getattr(s, "sub", "")) == "TiO2":
        return dict(none, status="not applicable")
    h = float(h_nom) if h_nom is not None else float("nan")
    ai = float(s.cfg.get("ai", 0.40))
    af = np.asarray(s.af, float) - float(eps)
    q = np.asarray(s.q, float)
    si = np.sin(np.radians(ai))
    q_top = k0 * (np.sqrt(max(si * si - sin2_ac, 1e-9)) + np.sqrt(max(np.sin(np.radians(FRINGE_AF[1])) ** 2 - sin2_ac, 1e-9)))
    if not np.isfinite(h) or FRINGE_D[1] * h < 2 * np.pi / q_top:
        return dict(none, status="too thin")
    ok = w["mask"] & ((q >= q_band[0]) & (q < q_band[1]))[None, None, :]
    E = np.where(ok, w["E"], 0.0).sum((0, 2))
    Y = np.where(ok, w["Y"], 0.0).sum((0, 2))
    P = np.where(ok, w["cfac"] * s.I_pre * w["E"], 0.0).sum((0, 2))
    sel = (af >= FRINGE_AF[0]) & (af <= FRINGE_AF[1]) & (E > 0)
    good = sel & (Y - P > 3.0 * np.sqrt(Y + 1.0))
    n = int(good.sum())
    if n < FRINGE_CELLS:
        return dict(none, status="no signal", n_af=n)
    M = (Y[good] - P[good]) / E[good]
    wt = np.minimum((Y[good] - P[good]) / np.sqrt(Y[good] + 1.0), 20.0)
    L = np.log(M)
    sa = np.sin(np.radians(af[good]))
    X0 = np.c_[np.ones(n), np.log(k0 * (si + sa))]
    r0 = L - X0 @ np.linalg.lstsq(X0 * wt[:, None], L * wt, rcond=None)[0]
    ss0 = float(np.sum((wt * r0) ** 2))
    dg = np.geomspace(max(FRINGE_D[0] * h, 2.0), FRINGE_D[1] * h, 160)
    best = None
    for i, d in enumerate(dg):
        s2 = min(1.0, h / d) * sin2_ac
        qp = k0 * (np.sqrt(max(si * si - s2, 1e-9)) + np.sqrt(np.maximum(sa * sa - s2, 1e-9)))
        for a in FRINGE_A:
            X = np.c_[X0, np.log(1 + a * a - 2 * a * np.cos(qp * d))]
            c = np.linalg.lstsq(X * wt[:, None], L * wt, rcond=None)[0]
            if c[2] <= 0:
                continue
            ss = float(np.sum((wt * (L - X @ c)) ** 2))
            if best is None or ss < best[0]:
                best = (ss, i, float(a))
    if best is None or ss0 <= 0:
        return dict(none, status="no fringe", n_af=n)
    score = 1.0 - best[0] / ss0
    d = float(dg[best[1]])
    out = dict(d_nm=d, contrast=best[2], score=float(score), n_af=n)
    s2 = min(1.0, h / d) * sin2_ac
    if k0 * (np.sqrt(max(si * si - s2, 1e-9)) + np.sqrt(max(float(np.max(sa)) ** 2 - s2, 1e-9))) * d < 2 * np.pi:
        return dict(out, d_nm=None, contrast=None, status="too thin")
    if score < FRINGE_SCORE:
        return dict(out, d_nm=None, contrast=None, status="no fringe")
    if best[1] in (0, len(dg) - 1):
        return dict(out, status="at range limit")
    return dict(out, status="fringe")


YON_AF = (0.08, 0.60)
YON_ROWS = 12
YON_CONTRAST = 0.25


def yoneda(s, w, sin2_ac, eps=0.0, q_band=FRINGE_Q):
    'Exit angle of the maximum of the metal-induced rate of a window, read from the counts alone. Exit-angle profile'
    none = dict(af_deg=None, rel=None, contrast=None, n_af=0)
    if str(getattr(s, "sub", "")) == "TiO2":
        return dict(none, status="not applicable")
    af = np.asarray(s.af, float) - float(eps)
    q = np.asarray(s.q, float)
    ok = w["mask"] & ((q >= q_band[0]) & (q < q_band[1]))[None, None, :]
    E = np.where(ok, w["E"], 0.0).sum((0, 2))
    Y = np.where(ok, w["Y"], 0.0).sum((0, 2))
    P = np.where(ok, w["cfac"] * s.I_pre * w["E"], 0.0).sum((0, 2))
    sel = (af >= YON_AF[0]) & (af <= YON_AF[1]) & (E > 0)
    good = sel & (Y - P > 3.0 * np.sqrt(Y + 1.0))
    n = int(good.sum())
    if n < YON_ROWS:
        return dict(none, status="no signal", n_af=n)
    r = np.where(E > 0, (Y - P) / np.maximum(E, 1e-30), np.nan)
    idx = np.where(sel)[0]
    sm = np.full(len(af), np.nan)
    for i in idx:
        v = r[max(i - 1, idx[0]): min(i + 2, idx[-1] + 1)]
        sm[i] = np.nanmean(v) if np.isfinite(v).any() else np.nan
    if not np.isfinite(sm[idx]).any():
        return dict(none, status="no signal", n_af=n)
    k = int(idx[np.nanargmax(sm[idx])])
    pos = float(af[k])
    edge = k in (idx[0], idx[-1])
    if not edge and np.isfinite(sm[k - 1]) and np.isfinite(sm[k + 1]):
        den = sm[k - 1] - 2 * sm[k] + sm[k + 1]
        if den < 0:
            pos = float(af[k] + 0.5 * (sm[k - 1] - sm[k + 1]) / den * (af[k + 1] - af[k]))
    rel = float(np.sin(np.radians(pos)) ** 2 / sin2_ac) if sin2_ac > 0 else None
    low = sm[idx[:3]]
    contrast = float(1.0 - np.nanmean(low) / sm[k]) if np.isfinite(low).any() and sm[k] > 0 else 0.0
    if not edge and contrast < YON_CONTRAST:
        return dict(none, contrast=contrast, status="flat profile", n_af=n)
    return dict(af_deg=pos, rel=rel, contrast=contrast, status="at range limit" if edge else "maximum", n_af=n)
