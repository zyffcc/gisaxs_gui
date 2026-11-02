"""Composition identifiability ceiling on the fixed 68-curve DEV subset.

For every case and every one of the 34 compositions, fit the nonlinear shape
coordinates by bounded least squares with nonnegative amplitudes profiled out
(variable projection), from several Latin-hypercube starts. The GT composition
additionally gets one start at the true u (reported separately, never mixed
into the fair comparison). Likelihood uses the serialized measurement sigma.

Question answered: does the GT composition fit clearly better than every
composition that does not contain it, or do non-nested alternatives reach an
equally good chi2? No network, no TEST, no training.
"""
from __future__ import annotations

import os
if __name__ == '__main__':
    # Script entry only (python -m research.identifiability.run), never on import: one BLAS thread, set
    # before numpy loads its BLAS; the pool workers inherit the environment. Importing this module (the
    # fitting tools import COMBOS/coordinates from it) leaves os.environ alone.
    for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(_name, '1')

import argparse
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.stats import qmc

from research.multishape_v1 import physics

COMBOS = physics.COMBOS
FIT_ORDER, FIT_ANGLES = 32, 48


def dev_cases(dev, per_combo=2):
    """First per_combo rows of each class, the same rule as the 68-curve DEV subset."""
    rows = []
    for code in range(34):
        rows.extend(np.flatnonzero(dev['combo'] == code)[:per_combo].tolist())
    return rows


def coordinates(types):
    """Nonlinear u coordinates active for this composition (amplitudes/B/C excluded)."""
    active = physics.active_mask(np.asarray(types)).astype(bool)
    active[[6, 13, 20, 27, 28, 29]] = False
    return np.flatnonzero(active)


class Fit:
    """Variable projection with per-slot column caching for finite differences."""

    def __init__(self, q, y, sigma, types, order=FIT_ORDER, angles=FIT_ANGLES):
        self.q, self.y, self.sigma = q, y, sigma
        self.types = np.asarray(types, int)
        self.order, self.angles = order, angles
        self.coords = coordinates(types)
        self.nfev = 0

    def column(self, u, j):
        if j == 4:
            return np.ones_like(self.q)
        if j == 5:
            p = physics.decode(u)
            return 1/(1+(self.q/p['sigma_res'])**p['nu_res'])
        typ = self.types[j]
        if not typ:
            return np.zeros_like(self.q)
        p = physics.decode(u)
        return physics.np_particle(self.q, int(typ), p['R'][j], p['sigma_R'][j], p['h'][j], p['sigma_h'][j],
                                   p['D'][j], p['sigma_D'][j], order=self.order, angles=self.angles)

    def matrix(self, u):
        return np.stack([self.column(u, j) for j in range(6)], axis=1)

    def solve(self, m):
        design, response = m/self.sigma[:, None], self.y/self.sigma
        scale = np.linalg.norm(design, axis=0)
        valid = scale > 1e-25
        coef = np.zeros(6)
        if valid.any():
            ridge = 1e-8
            a = np.vstack((design[:, valid]/scale[valid], np.sqrt(ridge)*np.eye(valid.sum())))
            coef[valid] = nnls(a, np.r_[response, np.zeros(valid.sum())], maxiter=2000)[0]/scale[valid]
        prediction = m@coef
        return coef, prediction, (prediction-self.y)/self.sigma

    def slot_of(self, coordinate):
        return 5 if coordinate >= 28 else coordinate // 7

    def run(self, u0, max_nfev=60):
        u0 = np.asarray(u0, float).copy()
        cache = {}

        def state(x):
            key = x.tobytes()
            if key not in cache:
                u = u0.copy()
                u[self.coords] = x
                m = self.matrix(u)
                cache.clear()
                cache[key] = (u, m, *self.solve(m))
                self.nfev += 1
            return cache[key]

        def fun(x):
            return state(x)[4]

        def jac(x):
            u, m, _, _, r0 = state(x)
            out = np.empty((len(r0), len(x)))
            for k, c in enumerate(self.coords):
                step = 1e-4 if x[k] <= .5 else -1e-4
                u2 = u.copy()
                u2[c] += step
                m2 = m.copy()
                j = self.slot_of(c)
                m2[:, j] = self.column(u2, j)
                out[:, k] = (self.solve(m2)[2]-r0)/step
                self.nfev += 1
            return out

        result = least_squares(fun, u0[self.coords], jac=jac, bounds=(0., 1.), max_nfev=max_nfev,
                               x_scale='jac', ftol=1e-8, xtol=1e-8, gtol=1e-8)
        u, m, coef, prediction, residual = state(result.x)
        return dict(u=u, coef=coef, prediction=prediction, chi2=float(residual@residual),
                    status=int(result.status), nfev=self.nfev)


def lnrmse(a, b):
    return float(np.sqrt(np.mean((np.log(np.maximum(a, 1e-20))-np.log(np.maximum(b, 1e-20)))**2)))


def contains(big, small):
    """Multiset containment on canonical type vectors."""
    b, s = Counter(int(t) for t in big if t), Counter(int(t) for t in small if t)
    return all(b[t] >= n for t, n in s.items())


def task(args):
    row, combo, start_id, u0, data = args
    q, y, sigma, clean = data
    types = COMBOS[combo]
    started = time.perf_counter()
    fit = Fit(q, y, sigma, types)
    try:
        r = fit.run(u0)
    except Exception as error:  # a failed start is recorded, never silently dropped
        return dict(row=row, combo=combo, start=start_id, failed=f'{type(error).__name__}: {error}')
    # Re-evaluate the found u at the full physics quadrature with re-profiled amplitudes.
    full = Fit(q, y, sigma, types, order=physics.DEFAULT_ORDER, angles=physics.DEFAULT_ANGLES)
    coef, prediction, residual = full.solve(full.matrix(r['u']))
    return dict(row=row, combo=combo, start=start_id, u=r['u'].tolist(), coef=coef.tolist(),
                chi2_fit=r['chi2'], chi2=float(residual@residual), clean_lnrmse=lnrmse(prediction, clean),
                observed_points=int(len(y)), zero_particles=int(np.sum((coef[:4] == 0) & (types > 0))),
                status=r['status'], nfev=r['nfev'], seconds=time.perf_counter()-started)


def build_tasks(dev, rows, starts, seed):
    tasks = []
    for row in rows:
        mask = dev['mask'][row]
        data = tuple(dev[k][row, mask].astype(float) for k in ('q', 'observed', 'sigma', 'clean'))
        gt_combo = int(dev['combo'][row])
        for combo in range(34):
            draws = qmc.LatinHypercube(32, seed=seed+1000*row+combo).random(starts)
            for s, u0 in enumerate(draws):
                tasks.append((row, combo, s, u0, data))
            if combo == gt_combo:
                tasks.append((row, combo, -1, dev['u'][row].astype(float), data))
    return tasks


def _limit_threads():
    """One BLAS thread in this process (main() and every pool worker): oversubscribed BLAS threads make
    the pooled fits far slower. Covers a main() called from a process that already loaded numpy."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def main():
    _limit_threads()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cases', type=str, default='all', help="'all' or comma list of positions in the 68 subset")
    parser.add_argument('--starts', type=int, default=6)
    parser.add_argument('--combos', type=str, default='all', help="'all' or comma list, pilot only")
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--seed', type=int, default=20260929)
    args = parser.parse_args()
    with np.load(args.dev, allow_pickle=False) as f:
        dev = {k: f[k] for k in f.files}
    subset = dev_cases(dev)
    positions = range(len(subset)) if args.cases == 'all' else [int(v) for v in args.cases.split(',')]
    rows = [subset[p] for p in positions]
    tasks = build_tasks(dev, rows, args.starts, args.seed)
    if args.combos != 'all':
        keep = {int(v) for v in args.combos.split(',')}
        tasks = [t for t in tasks if t[1] in keep]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'PROTOCOL.json').write_text(json.dumps(dict(
        rows=rows, positions=list(positions), starts=args.starts, seed=args.seed, fit_order=FIT_ORDER,
        fit_angles=FIT_ANGLES, final_order=physics.DEFAULT_ORDER, final_angles=physics.DEFAULT_ANGLES,
        likelihood='chi2 with serialized measurement sigma; amplitudes/B/C by ridged NNLS',
        gt_start='start=-1 uses true u; reported separately from the fair LHS starts',
        physics=physics.VERSION, test_accessed=False, tasks=len(tasks)), indent=1))
    t0 = time.perf_counter()
    out_file = args.out/'fits.jsonl'
    with Pool(args.workers, initializer=_limit_threads) as pool, out_file.open('a') as sink:
        for i, result in enumerate(pool.imap_unordered(task, tasks, chunksize=1)):
            sink.write(json.dumps(result)+'\n')
            sink.flush()
            if i % 50 == 0:
                print(json.dumps(dict(done=i+1, total=len(tasks), seconds=round(time.perf_counter()-t0))), flush=True)
    (args.out/'DONE.json').write_text(json.dumps(dict(tasks=len(tasks), seconds=time.perf_counter()-t0)))


if __name__ == '__main__':
    main()
