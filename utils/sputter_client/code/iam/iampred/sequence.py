'Whole-scan refinement of a prediction: all usable windows of one scan are fitted together (predict --sequence).'

from __future__ import annotations

import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from . import registry as REG
from . import runner

DT = (-1.5, 1.5)
BIG = 1.0e4
HUBER = 2.5
SPAN = 0.5
T_MIN = 0.25
FAR = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)

_W = {}


def _open(args):
    'Worker: build the contexts of the windows this process owns (the raw bins are dropped afterwards).'
    cfg, scan, times = args
    from . import adapter as A
    sc = A.ScanContexts(cfg, scan, times)
    _W["ctx"] = {float(t): c for t, c in zip(times, sc.ctx)}
    s = sc.scan
    meta = dict(t_dep=float(s.t_dep), rate_h=float(s.rate_h),
                t_sput=(float(s.cfg["t_sput"]) if "t_sput" in s.cfg else None))
    _W["meta"] = meta
    wins = [dict(t=float(t), n=int(c.f.ok.sum()), t_mean=float(c.t_mean), h_nom=c.h_nom_ctx, names=list(c.names),
                 lb=np.asarray(c.f.lb, float), ub=np.asarray(c.f.ub, float), mass=len(c.pnames) > len(c.names))
            for t, c in zip(times, sc.ctx)]
    return wins, meta


def mass(meta, t_mean, lnkap, dt):
    'Mass thickness of a window from the scan-level mass factor and onset shift (law of joint_fit.h_of).'
    dep = max(t_mean - meta["t_dep"] - dt, 0.05)
    if meta["t_sput"] is not None:
        dep = min(dep, meta["t_sput"])
    return math.exp(lnkap) * meta["rate_h"] * dep


def _eval(args):
    'Worker: forward model of parameter sets of its windows.'
    scan, jobs, save, lite = args
    meta = _W["meta"]
    out = []
    for t, xs, sh in jobs:
        ctx = _W["ctx"][t]
        ok = ctx.f.ok
        n = int(ok.sum())
        m_ = len(xs)
        R = np.empty((m_, n)); rep = np.full((m_, len(ctx.cfg["reported"])), np.nan)
        d5 = np.full(m_, BIG); hh = np.full(m_, np.nan); good = np.zeros(m_, bool)
        for i in range(m_):
            try:
                x = np.asarray(xs[i], float)
                if sh is not None:
                    h = mass(meta, ctx.t_mean, sh[i][0], sh[i][1])
                    x = np.r_[x, math.log(h / max(ctx.h_nom_ctx, 0.002))]
                (mu, _, _), mo, p, h = ctx.render(x)
                v, d = ctx.dev5(mu)
                if not np.isfinite(v):
                    raise FloatingPointError("dev5 is not finite")
                R[i] = (np.sign(ctx.w["Y"] - mu * ctx.w["E"]) * np.sqrt(d))[ok] / math.sqrt(n)
                rep[i] = ctx.reported(mo, p, h); d5[i] = v; hh[i] = h; good[i] = True
                if save and i == 0:
                    np.savez_compressed(Path(save) / "mu" / f"{scan}_{t:g}_seq.npz", mu=mu.astype(np.float32))
            except (Exception, SystemExit):
                R[i] = math.sqrt(BIG / n)
        out.append((t, None if lite else R, rep, d5, hh, good))
    return out


def huber(u):
    a = np.abs(u)
    return np.where(a <= HUBER, u, np.sign(u) * np.sqrt(np.maximum(2.0 * HUBER * a - HUBER * HUBER, 0.0)))


def dhuber(u):
    a = np.abs(u)
    return np.where(a <= HUBER, 1.0, HUBER / np.sqrt(np.maximum(2.0 * HUBER * a - HUBER * HUBER, 1e-300)))


class Sequence:
    'The windows of one scan in persistent worker processes. cross(): forward model of complete parameter vectors'

    def __init__(self, cfg, scan, times, workers, data=None, shadow=None, log=print):
        import scipy.sparse as sp
        self.sp = sp
        self.scan, self.log = scan, log
        self.t = [float(t) for t in times]
        W = self.W = len(self.t)
        env = REG.scan_env(cfg, scan)
        env.update(runner.site_env(cfg, data, shadow))
        n = self.n = max(1, min(int(workers), W))
        self.owner = [k % n for k in range(W)]
        mpc = mp.get_context("spawn")
        self.pools = [mpc.Pool(1, initializer=runner._init, initargs=(env,)) for _ in range(n)]
        res = [self.pools[i].apply_async(_open, ((cfg, scan, [self.t[k] for k in range(W) if self.owner[k] == i]),))
               for i in range(n)]
        info = {}
        for r in res:
            wins, self.meta = r.get()
            info.update({w["t"]: w for w in wins})
        self.win = [info[t] for t in self.t]
        nf = self.nf = len(self.win[0]["names"])
        mf = cfg.get("mass_factor") if self.win[0]["mass"] else None
        ns = self.ns = 2 if mf else 0
        self.lb = np.concatenate([w["lb"] for w in self.win] + ([[mf[0], DT[0]]] if ns else []))
        self.ub = np.concatenate([w["ub"] for w in self.win] + ([[mf[1], DT[1]]] if ns else []))
        self.rows = np.cumsum([0] + [w["n"] for w in self.win])
        self.tm = np.array([w["t_mean"] for w in self.win], float)
        self.tx = np.log(np.maximum(self.tm - self.meta["t_dep"], T_MIN))
        self.mono, self.smooth = [], []
        self.sl = self.ss = self.sf = 0.0
        self.tri = np.zeros(0, int); self.ta = np.zeros(0); self.tw = np.ones(0)
        self.fi = self.fj = np.zeros(0, int)
        self.z0 = None
        self._z = None
        self.njac = 0
        self.t0 = time.time()

    def close(self):
        for p in self.pools:
            p.close()
        for p in self.pools:
            p.join()

    def setup(self, x0, reps, mono, tol, weight, smooth=(), smooth_weight=0.0, far_weight=0.0):
        'x0: one start vector per window (pnames order). mono: names that may not decrease (scales tol, weight).'
        W, nf, ns = self.W, self.nf, self.ns
        x0 = [np.asarray(x, float) for x in x0]
        z0 = np.concatenate([x[:nf] for x in x0] + ([[float(np.median([x[nf] for x in x0])), 0.0]] if ns else []))
        self.z0 = np.clip(z0, self.lb + 1e-9, self.ub - 1e-9)
        idx = {r["name"]: k for k, r in enumerate(reps)}
        self.mono = [(idx[m], float(tol.get(m, 0.05)), m) for m in mono if m in idx]
        self.sl = math.sqrt(weight)
        self.sf = math.sqrt(max(float(far_weight), 0.0))
        pairs = [(i, i + o) for o in FAR for i in range(W - o)] if self.sf > 0 and self.mono else []
        self.fi = np.array([i for i, _ in pairs], int)
        self.fj = np.array([j for _, j in pairs], int)
        self.ss = math.sqrt(max(float(smooth_weight), 0.0))
        self.smooth = [(idx[m], float(sc), m) for m, sc in smooth if m in idx and sc > 0] if self.ss > 0 else []
        k = np.arange(1, W - 1)
        span = self.tx[k + 1] - self.tx[k - 1] if len(k) else np.zeros(0)
        keep = span > 1e-9
        if self.meta["t_sput"] is not None and len(k):
            te = self.meta["t_dep"] + self.meta["t_sput"]
            keep &= ~((self.tm[k - 1] < te) & (self.tm[k + 1] >= te))
        k, span = k[keep], span[keep]
        self.tri = k
        self.ta = (self.tx[k + 1] - self.tx[k]) / span if len(k) else np.zeros(0)
        self.tw = np.maximum(1.0, span / SPAN)
        self._z = None
        self.njac = 0
        self.t0 = time.time()

    def split(self, z):
        W, nf = self.W, self.nf
        return z[:W * nf].reshape(W, nf), (z[W * nf:] if self.ns else None)

    def _run(self, jobs, save=None, lite=False):
        'jobs {window k: (xs (m, nf), shared (m, 2) | None)} -> {k: (residual, reported, dev5, h_m, ok)}.'
        res = []
        for i, pool in enumerate(self.pools):
            mine = [(self.t[k], jobs[k][0], jobs[k][1]) for k in sorted(jobs) if self.owner[k] == i]
            if mine:
                res.append(pool.apply_async(_eval, ((self.scan, mine, save, lite),)))
        got = {}
        for r in res:
            for t, *v in r.get():
                got[t] = v
        return {k: got[self.t[k]] for k in jobs}

    def cross(self, jobs):
        'jobs {window k: complete parameter vectors (m, len(pnames))}, e.g. the solutions of other windows'
        res = self._run({k: (np.asarray(x, float), None) for k, x in jobs.items()}, lite=True)
        return {k: tuple(v[1:]) for k, v in res.items()}

    def _base(self, z, save=None):
        if save is None and self._z is not None and np.array_equal(z, self._z):
            return
        xs, sh = self.split(z)
        res = self._run({k: (xs[k][None], None if sh is None else sh[None]) for k in range(self.W)}, save)
        self._z = z.copy()
        self._R = [res[k][0][0] for k in range(self.W)]
        self._rep = np.stack([res[k][1][0] for k in range(self.W)])
        self._d5 = np.array([res[k][2][0] for k in range(self.W)])
        self._h = np.array([res[k][3][0] for k in range(self.W)])

    def decrease(self, rep, q):
        d = rep[:-1, q] - rep[1:, q]
        return np.where(np.isfinite(d), d, 0.0)

    def curve(self, rep, q):
        'Quantity q at the interior windows minus the value interpolated between the two neighbours (nan kept).'
        k = self.tri
        return rep[k, q] - self.ta * rep[k - 1, q] - (1.0 - self.ta) * rep[k + 1, q]

    def pen_mono(self, rep):
        return np.concatenate([self.sl * np.maximum(self.decrease(rep, q), 0.0) / sc for q, sc, _ in self.mono]
                              + [np.zeros(0)])

    def far(self, rep, q, sc):
        'Quantity q of an earlier window minus its value FAR places later, beyond one scale (<= 0: no fall).'
        d = rep[self.fi, q] - rep[self.fj, q] - sc
        return np.where(np.isfinite(d), d, 0.0)

    def pen_far(self, rep):
        if not len(self.fi):
            return np.zeros(0)
        return np.concatenate([self.sf * np.maximum(self.far(rep, q, sc), 0.0) / sc for q, sc, _ in self.mono])

    def pen_smooth(self, rep):
        out = [np.zeros(0)]
        for q, sc, _ in self.smooth:
            r = self.curve(rep, q)
            out.append(self.ss * huber(np.where(np.isfinite(r), r, 0.0) / (sc * self.tw)))
        return np.concatenate(out)

    def fun(self, z):
        self._base(z)
        return np.concatenate(self._R + [self.pen_mono(self._rep), self.pen_far(self._rep), self.pen_smooth(self._rep)])

    def state(self):
        pm, pf, ps = self.pen_mono(self._rep), self.pen_far(self._rep), self.pen_smooth(self._rep)
        return dict(sum_dev5=float(self._d5.sum()), median_dev5=float(np.median(self._d5)),
                    penalty=float(pm @ pm + pf @ pf + ps @ ps), penalty_monotone=float(pm @ pm),
                    penalty_far=float(pf @ pf), penalty_smooth=float(ps @ ps))

    def jac(self, z):
        sp = self.sp
        self._base(z)
        W, nf, ns = self.W, self.nf, self.ns
        nc = nf + ns
        xs, sh = self.split(z)
        jobs, steps = {}, []
        for k in range(W):
            P = np.repeat(xs[k][None], nc, 0)
            S = np.repeat(sh[None], nc, 0) if ns else None
            hk = np.empty(nc)
            for j in range(nc):
                v, u = (xs[k][j], self.ub[k * nf + j]) if j < nf else (sh[j - nf], self.ub[W * nf + j - nf])
                h = 1e-3 * max(1.0, abs(v))
                if v + h > u:
                    h = -h
                hk[j] = h
                if j < nf:
                    P[j, j] += h
                else:
                    S[j, j - nf] += h
            jobs[k] = (P, S); steps.append(hk)
        res = self._run(jobs)
        data, indices, dR = [], [], []
        for k in range(W):
            R, rep, _, _, good = res[k]
            Jk = np.where(good[:, None], (R - self._R[k][None]) / steps[k][:, None], 0.0).T
            dk = (rep - self._rep[k][None]) / steps[k][:, None]
            dR.append(np.where(np.isfinite(dk) & good[:, None], dk, 0.0))
            cols = np.r_[k * nf + np.arange(nf), W * nf + np.arange(ns)]
            data.append(Jk.ravel()); indices.append(np.tile(cols, Jk.shape[0]))
        ntot = int(self.rows[-1])
        J = sp.csr_matrix((np.concatenate(data), np.concatenate(indices).astype(np.int32), nc * np.arange(ntot + 1)),
                          shape=(ntot, W * nf + ns))
        rr, cc, vv = [], [], []
        row = 0
        loc, sha = np.arange(nf), W * nf + np.arange(ns)
        for q, sc, _ in self.mono:
            d = self.decrease(self._rep, q)
            for k in range(W - 1):
                if d[k] > 0:
                    c = self.sl / sc
                    rr += [row] * (2 * nf + ns)
                    cc += list(k * nf + loc) + list((k + 1) * nf + loc) + list(sha)
                    vv += list(c * dR[k][:nf, q]) + list(-c * dR[k + 1][:nf, q]) + list(c * (dR[k][nf:, q] - dR[k + 1][nf:, q]))
                row += 1
        if len(self.fi):
            for q, sc, _ in self.mono:
                d = self.far(self._rep, q, sc)
                c = self.sf / sc
                for i, j, dd in zip(self.fi, self.fj, d):
                    if dd > 0:
                        rr += [row] * (2 * nf + ns)
                        cc += list(i * nf + loc) + list(j * nf + loc) + list(sha)
                        vv += list(c * dR[i][:nf, q]) + list(-c * dR[j][:nf, q]) + list(c * (dR[i][nf:, q] - dR[j][nf:, q]))
                    row += 1
        for q, sc, _ in self.smooth:
            r = self.curve(self._rep, q)
            fin = np.isfinite(r)
            g = self.ss * dhuber(np.where(fin, r, 0.0) / (sc * self.tw)) / (sc * self.tw)
            for i, k in enumerate(self.tri):
                if fin[i]:
                    a_, c = self.ta[i], g[i]
                    b_ = 1.0 - a_
                    rr += [row] * (3 * nf + ns)
                    cc += list((k - 1) * nf + loc) + list(k * nf + loc) + list((k + 1) * nf + loc) + list(sha)
                    vv += (list(-c * a_ * dR[k - 1][:nf, q]) + list(c * dR[k][:nf, q]) + list(-c * b_ * dR[k + 1][:nf, q])
                           + list(c * (dR[k][nf:, q] - a_ * dR[k - 1][nf:, q] - b_ * dR[k + 1][nf:, q])))
                row += 1
        if row:
            J = sp.vstack([J, sp.coo_matrix((vv, (rr, cc)), shape=(row, W * nf + ns))], format="csr")
        self.njac += 1
        s = self.state()
        self.log(f"sequence {self.scan}: iteration {self.njac}  sum dev5 {s['sum_dev5']:.2f}  median {s['median_dev5']:.2f}  "
                 f"penalty monotone {s['penalty_monotone']:.3f} "
                 + (f"far {s['penalty_far']:.3f} " if self.sf > 0 else "") + f"smooth {s['penalty_smooth']:.3f}"
                 + (f"  kappa {math.exp(sh[0]):.3f}  dt {sh[1]:+.2f} s" if ns else "") + f"  ({time.time() - self.t0:.0f} s)")
        if self.on_iter is not None:
            try:
                self.on_iter(self.njac, s, xs, time.time() - self.t0)
            except Exception:
                pass
        return J

    on_iter = None
    """Callable (iteration, state, per-window local parameter vectors, seconds) called after every iteration (the
    state of the current solution: self._rep, self._h and self._d5 hold its reported quantities, masses, dev5)."""

    def solve(self, nfev):
        from scipy.optimize import least_squares
        return least_squares(self.fun, self.z0, jac=self.jac, bounds=(self.lb, self.ub), method="trf",
                             tr_solver="lsmr", x_scale="jac", max_nfev=nfev, ftol=1e-5, xtol=1e-8)


def refine(sq, x0, reps, mono, tol, weight, smooth, smooth_weight, nfev, outdir, far_weight=0.0):
    'sq: Sequence of the time-ordered windows; x0: their start vectors (pnames order). Returns'
    t0 = time.time()
    sq.setup(x0, reps, mono, tol, weight, smooth, smooth_weight, far_weight)
    sq._base(sq.z0)
    start = sq.state()
    r = sq.solve(nfev)
    sq._base(r.x, save=str(outdir))
    end = sq.state()
    xs, sh = sq.split(r.x)
    wins = {}
    for k, t in enumerate(sq.t):
        x = xs[k] if sh is None else np.r_[xs[k], math.log(sq._h[k] / max(sq.win[k]["h_nom"], 0.002))]
        wins[t] = dict(dev5=float(sq._d5[k]), reported=sq._rep[k].tolist(), x=np.asarray(x, float).tolist(),
                       h_m=float(sq._h[k]))
    dec = {}
    for q, sc, name in sq.mono:
        d = sq.decrease(sq._rep, q)
        v = sq._rep[:, q]
        fall = np.fmax.accumulate(v) - v
        fall = np.where(np.isfinite(fall), fall, 0.0)
        dec[name] = dict(scale=sc, n_over_scale=int(np.sum(d > sc)), max=float(max(d.max(), 0.0)) if len(d) else 0.0,
                         max_below_earlier=float(fall.max()) if len(fall) else 0.0,
                         n_below_earlier_over_scale=int(np.sum(fall > sc)))
    far = None
    if len(sq.fi):
        far = dict(weight=float(far_weight), offsets=list(FAR), dead_zone_scales=1.0, n_pairs=int(len(sq.fi)), quantities={})
        for q, sc, name in sq.mono:
            d = sq.far(sq._rep, q, sc)
            far["quantities"][name] = dict(scale=sc, n_charged=int(np.sum(d > 0)), max_over=float(max(d.max(), 0.0)))
    smo = {}
    for q, sc, name in sq.smooth:
        c = sq.curve(sq._rep, q)
        fin = np.isfinite(c)
        smo[name] = dict(scale=sc, n=int(fin.sum()), n_over_scale=int(np.sum(np.abs(c[fin]) > (sc * sq.tw)[fin])),
                         max_abs=float(np.abs(c[fin]).max()) if fin.any() else 0.0)
    coupling = ["scan-level mass factor and onset shift"] if sh is not None else []
    if sq.mono:
        coupling.append("monotone " + ", ".join(m for _, _, m in sq.mono)
                        + (" (also against earlier windows)" if len(sq.fi) else ""))
    if sq.smooth:
        coupling.append("smooth in ln time: " + ", ".join(m for _, _, m in sq.smooth))
    summ = dict(n_windows=sq.W, workers=sq.n, max_nfev=int(nfev), nfev=int(r.nfev), njev=int(r.njev),
                status=int(r.status), message=str(r.message), weight=float(weight),
                monotone=[m for _, _, m in sq.mono], start=start, end=end, decreases=dec, far=far,
                smooth=dict(weight=float(smooth_weight), huber=HUBER, span=SPAN, n_triples=int(len(sq.tri)), quantities=smo),
                shared=(dict(kappa=float(math.exp(sh[0])), dt_s=float(sh[1]),
                             kappa_at_bound=bool(min(sh[0] - sq.lb[-2], sq.ub[-2] - sh[0]) < 1e-3),
                             dt_at_bound=bool(min(sh[1] - sq.lb[-1], sq.ub[-1] - sh[1]) < 1e-3))
                        if sh is not None else None),
                coupling=" + ".join(coupling) if coupling else "none",
                seconds=time.time() - t0)
    return wins, summ
